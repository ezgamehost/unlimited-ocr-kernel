use std::path::Path;
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext},
    decoder::{Decoder, cache::LayerCache},
    model::config::ModelConfig,
    tensor::Tensor,
};

#[test]
#[ignore = "requires pinned original weights"]
fn actual_expert_scheduling_preserves_results_with_small_scratch() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
    let decoder = Decoder::new(&weights, config).unwrap();
    let input = Tensor::new(
        vec![1, 1280],
        (0..1280).map(|i| (i as f32 * 0.01).sin()).collect(),
    )
    .unwrap();
    let mut outputs = Vec::new();
    for (threads, scratch) in [(1, 1 << 30), (8, 1 << 30), (8, 32768)] {
        let mut cpu = CpuContext::new(Backend::Auto, threads, scratch).unwrap();
        let mut cache = LayerCache::new(10, 128, 128).unwrap();
        outputs.push(decoder.layer(1, &input, &mut cache, &mut cpu).unwrap().data);
    }
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0], outputs[2]);
}

#[test]
#[ignore = "requires pinned original weights and exported decoder logit fixtures"]
fn actual_decoder_teacher_forced_logits_match_upstream() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let fixture = Checkpoint::open(Path::new(
        "artifacts/reference-logo-f32/decoder.safetensors",
    ))
    .unwrap();
    let integers = |name: &str| -> Vec<usize> {
        let view = fixture.tensor(name).unwrap();
        assert_eq!(view.dtype, uocr::checkpoint::DType::I64);
        view.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|bytes| usize::try_from(i64::from_le_bytes(*bytes)).unwrap())
            .collect()
    };
    let tokens = integers("decode_token_ids");
    let all_ids = integers("input_token_ids");
    let positions = integers("logits_positions");
    assert_eq!(tokens.len(), 300);
    let prefix = all_ids.len() - tokens.len();
    assert_eq!(prefix, 277);
    assert_eq!(&all_ids[prefix..], tokens.as_slice());
    assert_eq!(positions, [276, 277, 278, 404, 405, 406, 532, 533, 576]);
    let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
    let embeddings = fixture.tensor("input_embeddings").unwrap();
    assert_eq!(embeddings.shape, [1, all_ids.len(), config.hidden_size]);
    let input = Tensor::new(
        vec![prefix, config.hidden_size],
        embeddings.to_f32()[..prefix * config.hidden_size].to_vec(),
    )
    .unwrap();
    let expected_view = fixture.tensor("logits").unwrap();
    assert_eq!(expected_view.shape, [positions.len(), config.vocab_size]);
    let expected = expected_view.to_f32();
    let decoder = Decoder::new(&weights, config.clone()).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    let (mut state, first) = decoder.prefill(&input, &mut cpu).unwrap();
    let mut checks = Vec::new();
    let mut check_logits = |position: usize, actual: &Tensor| {
        let Ok(row) = positions.binary_search(&position) else {
            return;
        };
        assert_eq!(actual.shape, [1, config.vocab_size]);
        let expected = &expected[row * config.vocab_size..(row + 1) * config.vocab_size];
        assert!(actual.data.iter().chain(expected).all(|v| v.is_finite()));
        let maximum = actual
            .data
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        let rms = (actual
            .data
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / config.vocab_size as f64)
            .sqrt();
        let outside = actual
            .data
            .iter()
            .zip(expected)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4 + 1e-4 * b.abs())
            .count();
        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
                .unwrap()
                .0
        };
        let top1_equal = argmax(&actual.data) == argmax(expected);
        eprintln!(
            "{}",
            serde_json::json!({"kind":"teacher_forced_logits", "position":position,
            "max_abs":maximum,"rms":rms,"outside_existing_tolerance":outside,
            "actual_top1":argmax(&actual.data),"reference_top1":argmax(expected)})
        );
        checks.push((position, outside, top1_equal));
    };
    check_logits(prefix - 1, &first);
    for (step, token) in tokens.into_iter().enumerate() {
        let logits = decoder
            .step(u32::try_from(token).unwrap(), &mut state, &mut cpu)
            .unwrap();
        assert_eq!(state.position(), prefix + step + 1);
        check_logits(prefix + step, &logits);
    }
    assert_eq!(checks.len(), positions.len());
    assert!(
        checks
            .iter()
            .all(|(_, outside, equal)| *outside == 0 && *equal),
        "teacher-forced logit discrepancies: {checks:?}"
    );
}

#[test]
#[ignore = "diagnostic: isolate whole-prefill drift from output-head drift"]
fn prefill_logit_component_diagnostic() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let fixture = Checkpoint::open(Path::new(
        "artifacts/reference-logo-f32/decoder.safetensors",
    ))
    .unwrap();
    let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
    let decoder = Decoder::new(&weights, config.clone()).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    let prefix = 277;
    let width = config.hidden_size;
    let mut hidden = Tensor::new(
        vec![prefix, width],
        fixture.tensor("input_embeddings").unwrap().to_f32()[..prefix * width].to_vec(),
    )
    .unwrap();
    let report = |kind: &str, layer: usize, actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        let rms = (actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let outside = actual
            .iter()
            .zip(expected)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4 + 1e-4 * b.abs())
            .count();
        eprintln!(
            "{}",
            serde_json::json!({"kind":kind,"layer":layer,"max_abs":maximum,
            "rms":rms,"outside_existing_tolerance":outside})
        );
    };
    for index in 0..config.num_hidden_layers {
        let mut cache = LayerCache::new(10, 128, 128).unwrap();
        hidden = decoder.layer(index, &hidden, &mut cache, &mut cpu).unwrap();
        let expected = fixture
            .tensor(&format!("layer.{index}.hidden"))
            .unwrap()
            .to_f32();
        report(
            "cumulative_prefill_last_hidden",
            index,
            &hidden.data[(prefix - 1) * width..],
            &expected[(prefix - 1) * width..prefix * width],
        );
        let input_name = if index == 0 {
            "input_embeddings".to_owned()
        } else {
            format!("layer.{}.hidden", index - 1)
        };
        let reference_input = fixture.tensor(&input_name).unwrap().to_f32();
        let reference_input = Tensor::new(
            vec![prefix, width],
            reference_input[..prefix * width].to_vec(),
        )
        .unwrap();
        let mut independent_cache = LayerCache::new(10, 128, 128).unwrap();
        let independent = decoder
            .layer(index, &reference_input, &mut independent_cache, &mut cpu)
            .unwrap();
        report(
            "independent_prefill_last_hidden",
            index,
            &independent.data[(prefix - 1) * width..],
            &expected[(prefix - 1) * width..prefix * width],
        );
    }
    let expected_logits = fixture.tensor("logits").unwrap().to_f32();
    let expected_logits = &expected_logits[..config.vocab_size];
    let actual_logits = decoder.logits(&hidden, &mut cpu).unwrap();
    report(
        "cumulative_prefill_logits",
        11,
        &actual_logits.data,
        expected_logits,
    );
    let last = fixture.tensor("layer.11.hidden").unwrap().to_f32();
    let reference_last = Tensor::new(
        vec![1, width],
        last[(prefix - 1) * width..prefix * width].to_vec(),
    )
    .unwrap();
    let isolated_head = decoder.logits(&reference_last, &mut cpu).unwrap();
    report(
        "head_with_reference_hidden",
        11,
        &isolated_head.data,
        expected_logits,
    );
}

#[test]
#[ignore = "requires pinned original weights and exported decoder trace"]
fn actual_decoder_layers_match_upstream_prefill_and_decode() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let fixture = Checkpoint::open(Path::new(
        "artifacts/reference-logo-f32/decoder.safetensors",
    ))
    .unwrap();
    let config = ModelConfig::load(Path::new("artifacts/upstream/config.json")).unwrap();
    let decoder = Decoder::new(&weights, config.clone()).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 10, 1 << 30).unwrap();
    let prefix = 277;
    for (layer, packed) in [0, 1, 11]
        .into_iter()
        .flat_map(|layer| [(layer, false), (layer, true)])
    {
        let input = fixture
            .tensor(if layer == 0 {
                "input_embeddings"
            } else if layer == 1 {
                "layer.0.hidden"
            } else {
                "layer.10.hidden"
            })
            .unwrap()
            .to_f32();
        let expected = fixture
            .tensor(&format!("layer.{layer}.hidden"))
            .unwrap()
            .to_f32();
        let mut cache = LayerCache::new(10, 128, 128).unwrap();
        let x = Tensor::new(vec![prefix, 1280], input[..prefix * 1280].to_vec()).unwrap();
        let y = decoder.layer(layer, &x, &mut cache, &mut cpu).unwrap();
        check(&y.data, &expected[..prefix * 1280], layer, 0);
        if packed {
            cache.pack_heads().unwrap();
        }
        eprintln!("reference layer={layer} head_major={packed}");
        for pos in prefix..prefix + 300 {
            let x =
                Tensor::new(vec![1, 1280], input[pos * 1280..(pos + 1) * 1280].to_vec()).unwrap();
            let y = decoder.layer(layer, &x, &mut cache, &mut cpu).unwrap();
            check(&y.data, &expected[pos * 1280..(pos + 1) * 1280], layer, pos);
        }
    }
}

fn check(actual: &[f32], expected: &[f32], layer: usize, position: usize) {
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let rms = (actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / actual.len() as f32)
        .sqrt();
    if position == 0 || [277, 278, 404, 405, 406, 532, 533, 576].contains(&position) {
        eprintln!("layer {layer} position {position}: max_abs={max}, rms={rms}");
    }
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-4 + 1e-4 * b.abs()),
        "layer {layer} max={max}"
    );
}
