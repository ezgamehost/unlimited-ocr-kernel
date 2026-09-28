use std::path::Path;
use uocr::vision::{Vision, window_partition, window_unpartition};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext},
    preprocess::PreparedImage,
    tensor::Tensor,
};

#[test]
fn window_shapes_reject_overflow_before_allocation() {
    let one = Tensor::new(vec![1, 1, 1, 1], vec![1.]).unwrap();
    for size in [usize::MAX, 1usize << (usize::BITS / 2)] {
        assert!(window_partition(&one, size).is_err());
        assert!(window_unpartition(&one, size, size, 1).is_err());
    }
}

#[test]
fn unpartition_rejects_empty_window_storage() {
    for shape in [vec![0, 1, 1, 1], vec![1, 1, 1, 0]] {
        let empty = Tensor::new(shape, vec![]).unwrap();
        assert!(window_unpartition(&empty, 1, 1, 1).is_err());
    }
}

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
    for (i, (a, b)) in actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > 1e-4 + 1e-4 * b.abs())
        .take(12)
    {
        eprintln!("mismatch {name} index={i} actual={a} expected={b}");
    }

    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= 1e-4 + 1e-4 * b.abs()),
        "{name}: max={max}"
    );
}

#[test]
#[ignore = "requires English multi-crop upstream vision export"]
fn original_gundam_pixels_and_embeddings_match_rust() {
    use uocr::preprocess::{ImageMode, prepare};
    let image = prepare(
        Path::new("benchmarks/corpus/english.png"),
        ImageMode::Gundam,
    )
    .unwrap();
    let directory = Path::new("artifacts/reference-english-gundam-f32");
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("metadata.json")).unwrap()).unwrap();
    assert_eq!(
        image.grid,
        [
            metadata["crop_grid"][0].as_u64().unwrap() as usize,
            metadata["crop_grid"][1].as_u64().unwrap() as usize
        ]
    );
    assert_eq!(
        image.image_tokens().unwrap(),
        metadata["image_tokens"].as_u64().unwrap() as usize
    );
    let inputs = Checkpoint::open(&directory.join("preprocess.safetensors")).unwrap();
    for (name, tensor) in std::iter::once(("global".to_string(), &image.global)).chain(
        image
            .crops
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("crop.{i}"), t)),
    ) {
        let expected = inputs.tensor(&name).unwrap().to_f32();
        let pixels = tensor.shape[1] * tensor.shape[2];
        for p in 0..pixels {
            for c in 0..3 {
                assert_eq!(
                    tensor.data[3 * p + c],
                    expected[c * pixels + p],
                    "{name} pixel {p} channel {c}"
                );
            }
        }
    }
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let model = Vision::new(&weights).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 10, 1 << 30).unwrap();
    let actual = model.encode(&image, &mut cpu).unwrap();
    let reference = Checkpoint::open(&directory.join("vision.safetensors")).unwrap();
    compare(
        "gundam merged",
        &actual.data,
        &reference.tensor("image_embeddings").unwrap().to_f32(),
    );
}

#[test]
#[ignore = "requires English multi-crop upstream vision export"]
fn original_english_global_stages_match_rust() {
    let image = uocr::preprocess::prepare(
        Path::new("benchmarks/corpus/english.png"),
        uocr::preprocess::ImageMode::Base,
    )
    .unwrap();
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let model = Vision::new(&weights).unwrap();
    let reference = Checkpoint::open(Path::new(
        "artifacts/reference-english-gundam-f32/vision.safetensors",
    ))
    .unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 10, 1 << 30).unwrap();
    let (sam, clip, projected) = model.encode_view_debug(&image.global, &mut cpu).unwrap();
    let expected = reference.tensor("global.sam").unwrap().to_f32();
    let expected_hwc: Vec<f32> = (0..256)
        .flat_map(|p| (0..1024).map(move |c| (p, c)))
        .map(|(p, c)| expected[c * 256 + p])
        .collect();
    compare("english sam", &sam.data, &expected_hwc);
    compare(
        "english clip",
        &clip.data,
        &reference.tensor("global.clip").unwrap().to_f32(),
    );
    compare(
        "english projected",
        &projected.data,
        &reference.tensor("global.projected").unwrap().to_f32(),
    );
}
