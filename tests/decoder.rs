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
