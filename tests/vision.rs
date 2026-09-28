use std::path::Path;
use uocr::vision::{Vision, window_partition, window_unpartition};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext},
    preprocess::PreparedImage,
    tensor::Tensor,
};

#[test]
fn padded_window_partition_is_reversible() {
    let x = Tensor::new(vec![1, 3, 5, 2], (0..30).map(|x| x as f32).collect()).unwrap();
    let windows = window_partition(&x, 2).unwrap();
    assert_eq!(windows.shape, [6, 2, 2, 2]);
    let y = window_unpartition(&windows, 3, 5, 2).unwrap();
    assert_eq!(y.shape, x.shape);
    assert_eq!(y.data, x.data);
}

#[test]
#[ignore = "requires original model and exported vision fixtures"]
fn original_vision_features_match_rust() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let inputs = Checkpoint::open(Path::new(
        "artifacts/reference-logo-f32/preprocess.safetensors",
    ))
    .unwrap();
    let reference =
        Checkpoint::open(Path::new("artifacts/reference-logo-f32/vision.safetensors")).unwrap();
    let input = inputs.tensor("global").unwrap().to_f32();
    let mut data = vec![0.; input.len()];
    for p in 0..1024 * 1024 {
        for c in 0..3 {
            data[p * 3 + c] = input[c * 1024 * 1024 + p];
        }
    }
    let image = PreparedImage {
        global: Tensor::new(vec![1, 1024, 1024, 3], data).unwrap(),
        crops: vec![],
        grid: [1, 1],
    };
    let model = Vision::new(&weights).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 10, 1 << 30).unwrap();
    let (sam, clip, actual) = model.encode_debug(&image, &mut cpu).unwrap();
    let expected = reference.tensor("global.sam").unwrap().to_f32();
    let mut expected_hwc = vec![0.; expected.len()];
    for p in 0..256 {
        for c in 0..1024 {
            expected_hwc[p * 1024 + c] = expected[c * 256 + p];
        }
    }
    compare("sam", &sam.data, &expected_hwc);
    compare(
        "clip",
        &clip.data,
        &reference.tensor("global.clip").unwrap().to_f32(),
    );
    compare(
        "projected",
        &actual.data,
        &reference.tensor("image_embeddings").unwrap().to_f32(),
    );
}

fn compare(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
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
    eprintln!("vision {name}: max_abs={max}, rms={rms}");
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-4 + 1e-4 * b.abs()),
        "{name}: max={max}"
    );
}
