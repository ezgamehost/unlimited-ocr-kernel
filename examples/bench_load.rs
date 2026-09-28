//! Attribute startup costs without running OCR. All integrity checks remain enabled.
use std::{path::Path, time::Instant};
use uocr::{checkpoint::Checkpoint, cpu::quantized::QuantizedWeights, vision::Vision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let source = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))?;
    eprintln!("source_open_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let hash = source.sha256();
    eprintln!(
        "source_hash_seconds={:.6} sha256={hash}",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let quantized = QuantizedWeights::load(
        Path::new("artifacts/unlimited-ocr-q8-block32.safetensors"),
        &source,
    )?;
    eprintln!("overlay_load_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let vision = Vision::new(&source)?;
    eprintln!("vision_load_seconds={:.6}", start.elapsed().as_secs_f64());
    std::hint::black_box((&quantized, &vision));
    Ok(())
}
