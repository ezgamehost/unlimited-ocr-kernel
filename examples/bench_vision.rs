use std::{hint::black_box, path::Path, time::Instant};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext},
    preprocess::{ImageMode, prepare},
    vision::Vision,
};
fn main() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let model = Vision::new(&weights).unwrap();
    let mode = if std::env::args().any(|v| v == "--gundam") {
        ImageMode::Gundam
    } else {
        ImageMode::Base
    };
    let input = prepare(Path::new("benchmarks/corpus/english.png"), mode).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    for _ in 0..3 {
        let start = Instant::now();
        black_box(model.encode(&input, &mut cpu).unwrap());
        println!(
            "image_tokens={} seconds={:.6}",
            input.image_tokens().unwrap(),
            start.elapsed().as_secs_f64()
        );
    }
}
