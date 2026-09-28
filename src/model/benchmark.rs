//! Resident, fixed-work native decoder measurement; not an OCR quality test.
use super::*;
use crate::cpu::{Backend, quantized::QuantizationMode};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "requires original weights, INT8 overlay, and English vision reference"]
fn fixed_work_decoder_benchmark() {
    let overlay = std::env::var_os("UOCR_BENCH_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "artifacts/unlimited-ocr-q8-block32.safetensors".into());
    let model = Model::load_with_quantization(
        Path::new("artifacts/upstream"),
        Some(&overlay),
        QuantizationMode::DynamicActivations,
    )
    .unwrap();
    let reference = Checkpoint::open(Path::new(
        "artifacts/reference-english-gundam-f32/vision.safetensors",
    ))
    .unwrap();
    let features = reference.tensor("image_embeddings").unwrap();
    assert_eq!(features.shape, [1513, model.config.hidden_size]);
    let features = features.to_f32();
    let input = model.tokenizer.encode(&default_prompt(), 1513).unwrap();
    let recorded: serde_json::Value = serde_json::from_str(include_str!(
        "../../docs/benchmarks/2026-09-28-block-dequantize/block-dequantize-fresh/english.json"
    ))
    .unwrap();
    let tokens: Vec<u32> = serde_json::from_value(recorded["token_ids"].clone()).unwrap();
    assert_eq!(tokens.len(), 250);
    let decoder = Decoder::new(&model.checkpoint, model.config.clone())
        .unwrap()
        .with_quantization(model.quantized.as_ref().unwrap(), model.quantization_mode);
    let mut embeddings = decoder.embedding(&input.ids).unwrap();
    let width = model.config.hidden_size;
    for (row, &position) in input.image_positions.iter().enumerate() {
        embeddings.data[position * width..(position + 1) * width]
            .copy_from_slice(&features[row * width..(row + 1) * width]);
    }
    let threads = std::env::var("UOCR_BENCH_THREADS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("UOCR_BENCH_THREADS must be an integer")
        })
        .unwrap_or(8);
    let mut cpu = CpuContext::new(Backend::Auto, threads, 1 << 30).unwrap();
    for repetition in 0..4 {
        let start = Instant::now();
        let (mut state, _) = decoder.prefill(&embeddings, &mut cpu).unwrap();
        let prefill_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut checksum = 0u64;
        let mut digest = Sha256::new();
        for token in tokens.iter().cycle().take(512) {
            let logits = decoder.step(*token, &mut state, &mut cpu).unwrap();
            // Consume every output; the reported duration includes this checksum.
            for value in &logits.data {
                checksum = checksum.wrapping_add(u64::from(value.to_bits()));
                digest.update(value.to_bits().to_le_bytes());
            }
            std::hint::black_box(&logits);
        }
        println!(
            "{}",
            serde_json::json!({
                "benchmark": "native_fixed_work_decoder", "warmup": repetition == 0,
                "repetition": repetition, "threads": threads, "prefix_tokens": input.ids.len(),
                "decode_steps": 512, "prefill_seconds": prefill_seconds,
                "decode_with_checksum_seconds": start.elapsed().as_secs_f64(),
                "logits_checksum": checksum, "logits_sha256": format!("{:x}", digest.finalize()),
                "overlay": overlay, "precision": model.precision(),
                "input_policy": "recorded English tokens cycled; teacher forced; ignores EOS"
            })
        );
    }
}

/// Diagnostic only: vary the output head while keeping decoder hidden states fixed.
#[test]
#[ignore = "requires pinned model, overlays and table page"]
fn table_prompt_quantization_diagnostic() {
    use crate::{
        cpu::{ops, quantized::Int8Matrix},
        decoder::cache::LayerCache,
        tensor::Tensor,
    };
    let overlay = std::env::var_os("UOCR_BENCH_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| "artifacts/unlimited-ocr-q4-block32.safetensors".into());
    let model = Model::load_with_quantization(
        Path::new("artifacts/upstream"),
        Some(&overlay),
        QuantizationMode::DynamicActivations,
    )
    .unwrap();
    let q8 = Checkpoint::open(Path::new("artifacts/unlimited-ocr-q8-block32.safetensors")).unwrap();
    assert_eq!(
        q8.sha256().unwrap(),
        "b6afd272e6f19de970f97245c83b09e110a20acc8033f2853d0e6739e5fde441"
    );
    let w = q8.tensor("lm_head.weight").unwrap();
    let q8_head = Int8Matrix::from_parts(
        w.shape[0],
        w.shape[1],
        w.bytes.iter().map(|&v| v as i8).collect(),
        q8.tensor("lm_head.weight.q8_scales").unwrap().to_f32(),
    )
    .unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    let image =
        preprocess::prepare(Path::new("benchmarks/corpus/table.png"), ImageMode::Gundam).unwrap();
    let image_tokens = image.image_tokens().unwrap();
    let features = model.vision.encode(&image, &mut cpu).unwrap();
    let decoder = Decoder::new(&model.checkpoint, model.config.clone())
        .unwrap()
        .with_quantization(model.quantized.as_ref().unwrap(), model.quantization_mode);
    for prompt in ["<image>document parsing.", "<image>\ndocument parsing."] {
        let input = model.tokenizer.encode(prompt, image_tokens).unwrap();
        let mut hidden = decoder.embedding(&input.ids).unwrap();
        let width = model.config.hidden_size;
        for (row, &position) in input.image_positions.iter().enumerate() {
            hidden.data[position * width..(position + 1) * width]
                .copy_from_slice(&features.data[row * width..(row + 1) * width]);
        }
        for layer in 0..model.config.num_hidden_layers {
            let mut cache = LayerCache::new(
                model.config.num_attention_heads,
                width / model.config.num_attention_heads,
                model.config.sliding_window,
            )
            .unwrap();
            hidden = decoder.layer(layer, &hidden, &mut cache, &mut cpu).unwrap();
        }
        let last = Tensor::new(
            vec![1, width],
            hidden.data[hidden.data.len() - width..].to_vec(),
        )
        .unwrap();
        let norm = model
            .checkpoint
            .tensor("model.norm.weight")
            .unwrap()
            .to_f32();
        let normalized = ops::rms_norm(&last, &norm, model.config.rms_norm_eps).unwrap();
        let baseline = decoder.logits(&hidden, &mut cpu).unwrap();
        let q8_logits = q8_head.linear_dynamic_int8(&normalized, &mut cpu).unwrap();
        let bf16_logits = ops::linear_weight(
            &normalized,
            model.checkpoint.tensor("lm_head.weight").unwrap(),
            None,
            &mut cpu,
        )
        .unwrap();
        for (head, logits) in [
            ("overlay", baseline),
            ("q8_dynamic", q8_logits),
            ("bf16_fp32", bf16_logits),
        ] {
            assert!(logits.data.iter().all(|v| v.is_finite()));
            let mut ranked: Vec<_> = logits.data.iter().copied().enumerate().collect();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            println!(
                "{}",
                serde_json::json!({"diagnostic":"table_prompt_head", "overlay":overlay,
                "prompt":prompt,"prefix_tokens":input.ids.len(),"head":head,"top5":&ranked[..5],
                "eos_logit":logits.data[1],"det_logit":logits.data[128818]})
            );
        }
    }
}
