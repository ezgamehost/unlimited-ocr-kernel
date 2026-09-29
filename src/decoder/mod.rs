pub mod cache;
pub mod moe;
use crate::{
    checkpoint::Checkpoint,
    cpu::{CpuContext, ops},
    error::{OcrError, Result},
    model::config::ModelConfig,
    tensor::Tensor,
};
use cache::LayerCache;
use rayon::prelude::*;

pub struct Decoder<'a> {
    weights: &'a Checkpoint,
    quantized: Option<(
        &'a crate::cpu::quantized::QuantizedWeights,
        crate::cpu::quantized::QuantizationMode,
    )>,
    config: ModelConfig,
}
pub struct DecodeState {
    layers: Vec<LayerCache>,
}

impl DecodeState {
    pub fn position(&self) -> usize {
        self.layers.first().map_or(0, LayerCache::position)
    }
}

impl<'a> Decoder<'a> {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn new(weights: &'a Checkpoint, config: ModelConfig) -> Result<Self> {
        config.validate()?;
        let hidden = config.hidden_size;
        let expect = |name: &str, shape: &[usize]| -> Result<()> {
            let w = weights.tensor(name)?;
            if w.shape != shape {
                return Err(OcrError::Shape(format!(
                    "{name}: expected {shape:?}, got {:?}",
                    w.shape
                )));
            }
            Ok(())
        };
        expect("model.embed_tokens.weight", &[config.vocab_size, hidden])?;
        expect("lm_head.weight", &[config.vocab_size, hidden])?;
        expect("model.norm.weight", &[hidden])?;
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            for part in ["input_layernorm", "post_attention_layernorm"] {
                expect(&format!("{prefix}.{part}.weight"), &[hidden])?;
            }
            for part in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                expect(
                    &format!("{prefix}.self_attn.{part}.weight"),
                    &[hidden, hidden],
                )?;
            }
            let ffn = |name: &str, width: usize| -> Result<()> {
                for part in ["gate_proj", "up_proj"] {
                    expect(&format!("{name}.{part}.weight"), &[width, hidden])?;
                }
                expect(&format!("{name}.down_proj.weight"), &[hidden, width])
            };
            if layer < config.first_k_dense_replace {
                ffn(&format!("{prefix}.mlp"), config.intermediate_size)?;
            } else {
                expect(
                    &format!("{prefix}.mlp.gate.weight"),
                    &[config.n_routed_experts, hidden],
                )?;
                ffn(
                    &format!("{prefix}.mlp.shared_experts"),
                    config.moe_intermediate_size * config.n_shared_experts,
                )?;
                for expert in 0..config.n_routed_experts {
                    ffn(
                        &format!("{prefix}.mlp.experts.{expert}"),
                        config.moe_intermediate_size,
                    )?;
                }
            }
        }
        Ok(Self {
            weights,
            config,
            quantized: None,
        })
    }

    pub(crate) fn with_quantization(
        mut self,
        weights: &'a crate::cpu::quantized::QuantizedWeights,
        mode: crate::cpu::quantized::QuantizationMode,
    ) -> Self {
        self.quantized = Some((weights, mode));
        self
    }

    fn normalize(&self, x: &Tensor, weight: &[f32]) -> Result<Tensor> {
        // Preserve the validated quantized trajectory. Unquantized projections
        // use a more accurate reduction to limit accumulated prefill drift.
        if self.quantized.is_none() {
            ops::rms_norm_pairwise(x, weight, self.config.rms_norm_eps)
        } else {
            ops::rms_norm(x, weight, self.config.rms_norm_eps)
        }
    }

    fn projection(&self, x: &Tensor, name: &str, cpu: &mut CpuContext) -> Result<Tensor> {
        if let Some((weights, mode)) = self.quantized
            && let Some(matrix) = weights.get(name)
        {
            return if mode == crate::cpu::quantized::QuantizationMode::DynamicActivations
                && x.shape.first() == Some(&1)
            {
                matrix.linear_dynamic_int8(x, cpu)
            } else {
                matrix.linear(x, cpu)
            };
        }
        ops::linear_weight(x, self.weights.tensor(name)?, None, cpu)
    }

    fn qkv(
        &self,
        x: &Tensor,
        prefix: &str,
        cpu: &mut CpuContext,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let names = ["q", "k", "v"].map(|part| format!("{prefix}.self_attn.{part}_proj.weight"));
        let width = self.config.hidden_size;
        // Conservatively allow a full F16/F32 weight expansion and activation
        // scratch for each concurrent projection. Small budgets keep the old path.
        let scratch = width
            .checked_mul(width)
            .and_then(|matrix| width.checked_mul(5).and_then(|n| matrix.checked_add(n)))
            .and_then(|n| n.checked_mul(3));
        if x.shape.first() == Some(&1)
            && cpu.threads() >= 3
            && scratch.is_some_and(|n| cpu.check_elements(n).is_ok())
        {
            let mut q_cpu = cpu.serial_worker(3);
            let mut k_cpu = cpu.serial_worker(3);
            let mut v_cpu = cpu.serial_worker(3);
            let pool = std::sync::Arc::clone(&cpu.pool);
            let (q, (k, v)) = pool.install(|| {
                rayon::join(
                    || self.projection(x, &names[0], &mut q_cpu),
                    || {
                        rayon::join(
                            || self.projection(x, &names[1], &mut k_cpu),
                            || self.projection(x, &names[2], &mut v_cpu),
                        )
                    },
                )
            });
            return Ok((q?, k?, v?));
        }
        Ok((
            self.projection(x, &names[0], cpu)?,
            self.projection(x, &names[1], cpu)?,
            self.projection(x, &names[2], cpu)?,
        ))
    }

    fn feed_forward(&self, x: &Tensor, prefix: &str, cpu: &mut CpuContext) -> Result<Tensor> {
        let mut gate = self.projection(x, &format!("{prefix}.gate_proj.weight"), cpu)?;
        let up = self.projection(x, &format!("{prefix}.up_proj.weight"), cpu)?;
        ops::silu(&mut gate.data);
        for (value, u) in gate.data.iter_mut().zip(up.data) {
            *value *= u;
        }
        self.projection(&gate, &format!("{prefix}.down_proj.weight"), cpu)
    }

    fn experts(&self, x: &Tensor, prefix: &str, cpu: &mut CpuContext) -> Result<Tensor> {
        let hidden = self.config.hidden_size;
        let gates = self.projection(x, &format!("{prefix}.gate.weight"), cpu)?;
        let mut assignments = vec![Vec::<(usize, f32)>::new(); self.config.n_routed_experts];
        for (row, scores) in gates
            .data
            .chunks_exact(self.config.n_routed_experts)
            .enumerate()
        {
            for (expert, probability) in moe::route(
                scores,
                self.config.num_experts_per_tok,
                self.config.norm_topk_prob,
                self.config.routed_scaling_factor,
            )? {
                assignments[expert].push((row, probability));
            }
        }
        if x.shape[0] == 1 && cpu.threads() > 1 {
            let selected: Vec<_> = assignments
                .iter()
                .enumerate()
                .filter_map(|(expert, rows)| rows.first().map(|&(_, p)| (expert, p)))
                .collect();
            // Reserve conservatively for expanded F16/F32 matrices as well as
            // dynamic activation quantization. Low-memory requests keep the
            // serial-expert path instead of acquiring a stricter failure mode.
            let task_count = selected.len() + 1;
            let width = self
                .config
                .moe_intermediate_size
                .checked_mul(self.config.n_shared_experts.max(1));
            let scratch = width
                .and_then(|width| {
                    width
                        .checked_add(hidden)
                        .and_then(|n| n.checked_mul(3))
                        .and_then(|n| {
                            width
                                .checked_mul(hidden)
                                .and_then(|matrix| n.checked_add(matrix))
                        })
                })
                .and_then(|n| n.checked_mul(task_count));

            if !selected.is_empty() && scratch.is_some_and(|n| cpu.check_elements(n).is_ok()) {
                let pool = std::sync::Arc::clone(&cpu.pool);
                let outputs: Vec<_> = pool.install(|| {
                    (0..task_count)
                        .into_par_iter()
                        .map(|task| {
                            let mut worker = if task == 0 {
                                cpu.shared_expert_worker(task_count, self.config.n_shared_experts)
                            } else {
                                cpu.serial_worker(task_count)
                            };
                            let name = if task == 0 {
                                format!("{prefix}.shared_experts")
                            } else {
                                format!("{prefix}.experts.{}", selected[task - 1].0)
                            };
                            self.feed_forward(x, &name, &mut worker)
                        })
                        .collect()
                });
                let mut outputs = outputs.into_iter();
                let mut result = outputs.next().expect("shared expert task exists")?;
                // Indexed collection and ordered accumulation preserve the original
                // expert reduction order regardless of task completion order.
                for ((_, probability), output) in selected.into_iter().zip(outputs) {
                    for (dest, value) in result.data.iter_mut().zip(output?.data) {
                        *dest += probability * value;
                    }
                }
                return Ok(result);
            }
        }
        let mut result = self.feed_forward(x, &format!("{prefix}.shared_experts"), cpu)?;
        for (expert, rows) in assignments.into_iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let mut inputs = Vec::with_capacity(rows.len() * hidden);
            for &(row, _) in &rows {
                inputs.extend_from_slice(&x.data[row * hidden..(row + 1) * hidden]);
            }
            let input = Tensor::new(vec![rows.len(), hidden], inputs)?;
            let output = self.feed_forward(&input, &format!("{prefix}.experts.{expert}"), cpu)?;
            for ((row, probability), expert_output) in
                rows.into_iter().zip(output.data.chunks_exact(hidden))
            {
                for (dest, value) in result.data[row * hidden..(row + 1) * hidden]
                    .iter_mut()
                    .zip(expert_output)
                {
                    *dest += probability * value;
                }
            }
        }
        Ok(result)
    }

    pub fn embedding(&self, ids: &[u32]) -> Result<Tensor> {
        let w = self.weights.tensor("model.embed_tokens.weight")?;
        let width = self.config.hidden_size;
        let bytes_per_row = w.bytes.len() / self.config.vocab_size;
        let mut data = Vec::with_capacity(ids.len() * width);
        for &id in ids {
            if id as usize >= self.config.vocab_size {
                return Err(OcrError::Shape("token outside vocabulary".into()));
            }
            let row = crate::checkpoint::WeightView {
                dtype: w.dtype,
                shape: &[width],
                bytes: &w.bytes[id as usize * bytes_per_row..(id as usize + 1) * bytes_per_row],
            };
            data.extend(row.to_f32());
        }
        Tensor::new(vec![ids.len(), width], data)
    }

    pub fn layer(
        &self,
        index: usize,
        x: &Tensor,
        cache: &mut LayerCache,
        cpu: &mut CpuContext,
    ) -> Result<Tensor> {
        x.validate()?;
        let hidden = self.config.hidden_size;
        if index >= self.config.num_hidden_layers
            || x.shape.len() != 2
            || x.shape[1] != hidden
            || x.shape[0] == 0
            || (cache.position() != 0 && x.shape[0] != 1)
        {
            return Err(OcrError::Shape("invalid decoder layer input".into()));
        }
        let rows = x.shape[0];
        let position = cache.position();
        if position
            .checked_add(rows)
            .is_none_or(|n| n > self.config.max_position_embeddings)
        {
            return Err(OcrError::Resource("maximum context exceeded".into()));
        }
        let prefix = format!("model.layers.{index}");
        let norm = self
            .weights
            .tensor(&format!("{prefix}.input_layernorm.weight"))?
            .to_f32();
        let normalized = self.normalize(x, &norm)?;
        let (mut q, mut k, v) = self.qkv(&normalized, &prefix, cpu)?;
        let heads = self.config.num_attention_heads;
        let dim = hidden / heads;
        for row in 0..rows {
            ops::rope_pair(
                &mut q.data[row * hidden..(row + 1) * hidden],
                &mut k.data[row * hidden..(row + 1) * hidden],
                dim,
                position + row,
                self.config.rope_theta,
            )?;
        }
        if position == 0 {
            cache.prefill(&k.data, &v.data, rows)?;
        } else {
            cache.push(&k.data, &v.data)?;
        }
        let joined = if position > 0 {
            if cache.is_head_major() {
                ops::attention_decode_head_major(
                    &q,
                    cache.keys(),
                    cache.values(),
                    heads,
                    cache.tokens(),
                    cpu,
                )?
                .data
            } else {
                ops::attention_decode(&q, cache.keys(), cache.values(), heads, cpu)?.data
            }
        } else {
            let split = |data: &[f32], tokens: usize| -> Result<Tensor> {
                let mut out = vec![0.; data.len()];
                for head in 0..heads {
                    for token in 0..tokens {
                        out[(head * tokens + token) * dim..(head * tokens + token + 1) * dim]
                            .copy_from_slice(
                                &data[token * hidden + head * dim
                                    ..token * hidden + (head + 1) * dim],
                            );
                    }
                }
                Tensor::new(vec![heads, tokens, dim], out)
            };
            let attention = ops::attention(
                &split(&q.data, rows)?,
                &split(cache.keys(), cache.tokens())?,
                &split(cache.values(), cache.tokens())?,
                position == 0,
                cpu,
            )?;
            let mut joined = vec![0.; rows * hidden];
            for head in 0..heads {
                for row in 0..rows {
                    joined[row * hidden + head * dim..row * hidden + (head + 1) * dim]
                        .copy_from_slice(
                            &attention.data
                                [(head * rows + row) * dim..(head * rows + row + 1) * dim],
                        );
                }
            }
            joined
        };
        let mut residual = self.projection(
            &Tensor::new(vec![rows, hidden], joined)?,
            &format!("{prefix}.self_attn.o_proj.weight"),
            cpu,
        )?;
        for (dest, input) in residual.data.iter_mut().zip(&x.data) {
            *dest += input;
        }
        let postnorm = self
            .weights
            .tensor(&format!("{prefix}.post_attention_layernorm.weight"))?
            .to_f32();
        let normalized = self.normalize(&residual, &postnorm)?;
        let mlp = if index < self.config.first_k_dense_replace {
            self.feed_forward(&normalized, &format!("{prefix}.mlp"), cpu)?
        } else {
            self.experts(&normalized, &format!("{prefix}.mlp"), cpu)?
        };
        for (dest, value) in residual.data.iter_mut().zip(mlp.data) {
            *dest += value;
        }
        Ok(residual)
    }

    pub fn logits(&self, hidden: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        let width = self.config.hidden_size;
        if hidden.data.len() < width {
            return Err(OcrError::Shape("missing final hidden state".into()));
        }
        let last = Tensor::new(
            vec![1, width],
            hidden.data[hidden.data.len() - width..].to_vec(),
        )?;
        let norm = self.weights.tensor("model.norm.weight")?.to_f32();
        let normalized = self.normalize(&last, &norm)?;
        self.projection(&normalized, "lm_head.weight", cpu)
    }

    pub fn prefill(
        &self,
        embeddings: &Tensor,
        cpu: &mut CpuContext,
    ) -> Result<(DecodeState, Tensor)> {
        let mut hidden = embeddings.clone();
        let mut layers = Vec::with_capacity(self.config.num_hidden_layers);
        for index in 0..self.config.num_hidden_layers {
            let mut cache = LayerCache::new(
                self.config.num_attention_heads,
                self.config.hidden_size / self.config.num_attention_heads,
                self.config.sliding_window,
            )?;
            hidden = self.layer(index, &hidden, &mut cache, cpu)?;
            cache.pack_heads()?;
            layers.push(cache);
        }
        Ok((DecodeState { layers }, self.logits(&hidden, cpu)?))
    }

    pub fn step(
        &self,
        token: u32,
        state: &mut DecodeState,
        cpu: &mut CpuContext,
    ) -> Result<Tensor> {
        cpu.run(|cpu| self.step_on_worker(token, state, cpu))
    }

    fn step_on_worker(
        &self,
        token: u32,
        state: &mut DecodeState,
        cpu: &mut CpuContext,
    ) -> Result<Tensor> {
        if state.layers.len() != self.config.num_hidden_layers {
            return Err(OcrError::Shape("decoder state mismatch".into()));
        }
        let mut hidden = self.embedding(&[token])?;
        for (index, cache) in state.layers.iter_mut().enumerate() {
            hidden = self.layer(index, &hidden, cache, cpu)?;
        }
        self.logits(&hidden, cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires original weights and upstream teacher-forced routing trace"]
    fn actual_teacher_forced_expert_routes_match_upstream() {
        use crate::{checkpoint::DType, cpu::Backend};
        use std::path::Path;
        let weights = Checkpoint::open(Path::new(
            "artifacts/upstream/model-00001-of-000001.safetensors",
        ))
        .unwrap();
        let fixture = Checkpoint::open(Path::new(
            "artifacts/reference-logo-f32/decoder.safetensors",
        ))
        .unwrap();
        let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.first_k_dense_replace, 1);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.n_routed_experts, 64);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.num_attention_heads, 10);
        assert_eq!(config.hidden_size, 1280);
        let decoder = Decoder::new(&weights, config.clone()).unwrap();
        let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
        let (prefix, positions) = (277usize, 577usize);
        let view = fixture.tensor("decode_token_ids").unwrap();
        assert_eq!(view.dtype, DType::I64);
        let tokens: Vec<u32> = view
            .bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u32::try_from(i64::from_le_bytes(*b)).unwrap())
            .collect();
        assert_eq!(tokens.len(), positions - prefix);
        let embeddings = fixture.tensor("input_embeddings").unwrap();
        assert_eq!(embeddings.shape, [1, positions, config.hidden_size]);
        let x = Tensor::new(
            vec![prefix, config.hidden_size],
            embeddings.to_f32()[..prefix * config.hidden_size].to_vec(),
        )
        .unwrap();
        let ((mut state, _), prefill_routes) = cpu.run(|cpu| {
            let (result, routes) = moe::capture_routes(|| decoder.prefill(&x, cpu));
            (result.unwrap(), routes)
        });
        assert_eq!(state.position(), prefix);
        assert!(state.layers.iter().all(LayerCache::is_head_major));
        let routed_layers = config.num_hidden_layers - 1;
        assert_eq!(prefill_routes.len(), routed_layers * prefix);
        let mut actual: Vec<Vec<Vec<(usize, f32)>>> = (0..routed_layers)
            .map(|_| Vec::with_capacity(positions))
            .collect();
        for (i, row) in prefill_routes.into_iter().enumerate() {
            actual[i / prefix].push(row);
        }
        for (step, token) in tokens.into_iter().enumerate() {
            let routes = cpu.run(|cpu| {
                let (result, routes) = moe::capture_routes(|| decoder.step(token, &mut state, cpu));
                result.unwrap();
                routes
            });
            assert_eq!(state.position(), prefix + step + 1);
            assert!(state.layers.iter().all(|cache| cache.is_head_major()
                && cache.tokens() == prefix + (step + 1).min(config.sliding_window)));
            assert_eq!(routes.len(), routed_layers);
            for (layer, row) in routes.into_iter().enumerate() {
                actual[layer].push(row);
            }
        }
        let mut total_mismatches = 0;
        for (layer_index, actual) in actual.into_iter().enumerate() {
            let layer = layer_index + 1;
            assert_eq!(actual.len(), positions);
            let ids = fixture
                .tensor(&format!("layer.{layer}.expert_ids"))
                .unwrap();
            assert_eq!(ids.dtype, DType::I64);
            assert_eq!(ids.shape, [positions, config.num_experts_per_tok]);
            let ids: Vec<usize> = ids
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| usize::try_from(i64::from_le_bytes(*b)).unwrap())
                .collect();
            let probabilities = fixture
                .tensor(&format!("layer.{layer}.expert_weights"))
                .unwrap();
            assert_eq!(probabilities.dtype, DType::F32);
            assert_eq!(probabilities.shape, [positions, config.num_experts_per_tok]);
            let probabilities = probabilities.to_f32();
            let (mut id_mismatches, mut weight_mismatches, mut max_error) = (0, 0, 0f32);
            for (position, mut actual) in actual.into_iter().enumerate() {
                let begin = position * config.num_experts_per_tok;
                let mut expected: Vec<_> = ids[begin..begin + config.num_experts_per_tok]
                    .iter()
                    .copied()
                    .zip(
                        probabilities[begin..begin + config.num_experts_per_tok]
                            .iter()
                            .copied(),
                    )
                    .collect();
                assert_eq!(actual.len(), expected.len());
                actual.sort_unstable_by_key(|&(id, _)| id);
                expected.sort_unstable_by_key(|&(id, _)| id);
                assert!(expected.windows(2).all(|w| w[0].0 != w[1].0));
                for ((actual_id, a), (expected_id, e)) in actual.into_iter().zip(expected) {
                    assert!(
                        actual_id < config.n_routed_experts
                            && expected_id < config.n_routed_experts
                    );
                    assert!(a.is_finite() && e.is_finite() && a >= 0. && e >= 0.);
                    if actual_id != expected_id {
                        id_mismatches += 1;
                        continue;
                    }
                    let error = (a - e).abs();
                    max_error = max_error.max(error);
                    if error > 1e-5 + 1e-4 * e.abs() {
                        weight_mismatches += 1;
                    }
                }
            }
            println!(
                "{}",
                serde_json::json!({"gate":"teacher_forced_expert_routes","layer":layer,"positions":positions,"id_mismatches":id_mismatches,"weight_tolerance_violations":weight_mismatches,"max_weight_error":max_error})
            );
            total_mismatches += id_mismatches + weight_mismatches;
        }
        assert_eq!(
            total_mismatches, 0,
            "teacher-forced expert routing differs from upstream"
        );
    }

    #[test]
    #[ignore = "requires original weights and upstream expert ID/weight trace"]
    fn actual_expert_routes_match_upstream_all_moe_layers() {
        use crate::{checkpoint::DType, cpu::Backend};
        use std::path::Path;
        let weights = Checkpoint::open(Path::new(
            "artifacts/upstream/model-00001-of-000001.safetensors",
        ))
        .unwrap();
        let fixture = Checkpoint::open(Path::new(
            "artifacts/reference-logo-f32/decoder.safetensors",
        ))
        .unwrap();
        let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.first_k_dense_replace, 1);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.n_routed_experts, 64);
        assert_eq!(config.sliding_window, 128);
        assert_eq!(config.num_attention_heads, 10);
        assert_eq!(config.hidden_size, 1280);
        let decoder = Decoder::new(&weights, config.clone()).unwrap();
        let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
        let (prefix, positions) = (277usize, 577usize);
        assert_eq!(config.num_experts_per_tok, 6);
        for layer in 1..config.num_hidden_layers {
            let input = fixture
                .tensor(&format!("layer.{}.hidden", layer - 1))
                .unwrap();
            assert_eq!(input.shape, [1, positions, config.hidden_size]);
            let input = input.to_f32();
            let ids = fixture
                .tensor(&format!("layer.{layer}.expert_ids"))
                .unwrap();
            assert_eq!(ids.dtype, DType::I64);
            assert_eq!(ids.shape, [positions, config.num_experts_per_tok]);
            let ids: Vec<usize> = ids
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| usize::try_from(i64::from_le_bytes(*b)).unwrap())
                .collect();
            let probabilities = fixture
                .tensor(&format!("layer.{layer}.expert_weights"))
                .unwrap();
            assert_eq!(probabilities.dtype, DType::F32);
            assert_eq!(probabilities.shape, [positions, config.num_experts_per_tok]);
            let probabilities = probabilities.to_f32();
            let mut cache = LayerCache::new(
                config.num_attention_heads,
                config.hidden_size / config.num_attention_heads,
                config.sliding_window,
            )
            .unwrap();
            let mut actual = Vec::with_capacity(positions);
            for (start, count) in
                std::iter::once((0, prefix)).chain((prefix..positions).map(|p| (p, 1)))
            {
                let x = Tensor::new(
                    vec![count, config.hidden_size],
                    input[start * config.hidden_size..(start + count) * config.hidden_size]
                        .to_vec(),
                )
                .unwrap();
                let captured = cpu.run(|cpu| {
                    let (result, routes) =
                        moe::capture_routes(|| decoder.layer(layer, &x, &mut cache, cpu));
                    result.unwrap();
                    routes
                });
                assert_eq!(captured.len(), count, "layer={layer} start={start}");
                actual.extend(captured);
                if start == 0 {
                    cache.pack_heads().unwrap();
                }
                assert!(cache.is_head_major());
                assert_eq!(cache.position(), start + count);
                assert_eq!(
                    cache.tokens(),
                    prefix + (start + count - prefix).min(config.sliding_window)
                );
            }
            assert_eq!(actual.len(), positions);
            let mut max_error = 0f32;
            for (position, mut actual) in actual.into_iter().enumerate() {
                let begin = position * config.num_experts_per_tok;
                let mut expected: Vec<_> = ids[begin..begin + config.num_experts_per_tok]
                    .iter()
                    .copied()
                    .zip(
                        probabilities[begin..begin + config.num_experts_per_tok]
                            .iter()
                            .copied(),
                    )
                    .collect();
                assert_eq!(actual.len(), expected.len());
                actual.sort_unstable_by_key(|&(id, _)| id);
                expected.sort_unstable_by_key(|&(id, _)| id);
                assert!(expected.windows(2).all(|w| w[0].0 != w[1].0));
                for ((actual_id, a), (expected_id, e)) in actual.into_iter().zip(expected) {
                    assert!(expected_id < config.n_routed_experts);
                    assert_eq!(actual_id, expected_id, "layer={layer} position={position}");
                    assert!(a.is_finite() && e.is_finite() && a >= 0. && e >= 0.);
                    let error = (a - e).abs();
                    max_error = max_error.max(error);
                    assert!(
                        error <= 1e-5 + 1e-4 * e.abs(),
                        "layer={layer} position={position} expert={actual_id} actual={a} expected={e}"
                    );
                }
            }
            println!(
                "{}",
                serde_json::json!({"gate":"expert_routes", "layer":layer,"positions":positions,"prefill_positions":prefix,"decode_positions":positions-prefix,"selected_ids_exact":true,"max_weight_error":max_error})
            );
        }
    }

    #[test]
    #[ignore = "requires pinned original model weights"]
    fn qkv_matches_independent_projections_across_worker_and_memory_budgets() {
        use crate::cpu::Backend;
        use std::path::Path;
        let weights = Checkpoint::open(Path::new(
            "artifacts/upstream/model-00001-of-000001.safetensors",
        ))
        .unwrap();
        let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
        let decoder = Decoder::new(&weights, config).unwrap();
        for rows in [1, 2] {
            let x = Tensor::new(
                vec![rows, 1280],
                (0..rows * 1280).map(|i| (i as f32 * 0.01).sin()).collect(),
            )
            .unwrap();
            let mut serial = CpuContext::new(Backend::Auto, 1, 1 << 30).unwrap();
            let expected: Vec<_> = ["q", "k", "v"]
                .iter()
                .map(|part| {
                    decoder
                        .projection(
                            &x,
                            &format!("model.layers.1.self_attn.{part}_proj.weight"),
                            &mut serial,
                        )
                        .unwrap()
                })
                .collect();
            for (threads, budget) in [
                (1, 1 << 30),
                (2, 1 << 30),
                (3, 1 << 30),
                (4, 1 << 30),
                (8, 1 << 30),
                (8, 32768),
            ] {
                if rows > 1 && budget == 32768 {
                    continue;
                }
                let mut cpu = CpuContext::new(Backend::Auto, threads, budget).unwrap();
                let (q, k, v) = cpu
                    .run(|cpu| decoder.qkv(&x, "model.layers.1", cpu))
                    .unwrap();
                for (actual, expected) in [q, k, v].into_iter().zip(&expected) {
                    assert_eq!(actual.shape, expected.shape);
                    assert_eq!(
                        actual.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected
                            .data
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "rows={rows}, threads={threads}, budget={budget}"
                    );
                }
            }
        }
    }
}
