//! Attribute startup costs without running OCR. All integrity checks remain enabled.
use std::{path::Path, time::Instant};
use uocr::{checkpoint::Checkpoint, cpu::quantized::QuantizedWeights, vision::Vision};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|v| v == "--stream-hashes") {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut buffer = vec![0u8; 1 << 20];
        for path in [
            "artifacts/upstream/model-00001-of-000001.safetensors",
            "artifacts/unlimited-ocr-q8-block32.safetensors",
        ] {
            let start = Instant::now();
            let mut file = std::fs::File::open(path)?;
            let mut hash = Sha256::new();
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            eprintln!(
                "stream_hash_seconds={:.6} path={path} sha256={:x}",
                start.elapsed().as_secs_f64(),
                hash.finalize()
            );
        }
        return Ok(());
    }
    let start = Instant::now();
    let source = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))?;
    eprintln!("source_open_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let hash = source.sha256()?;
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
