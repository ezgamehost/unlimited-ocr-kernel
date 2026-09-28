//! Resident, fixed-work native decoder measurement; not an OCR quality test.
use super::*;
use crate::cpu::{Backend, quantized::QuantizationMode};

#[test]
#[ignore = "requires original weights, INT8 overlay, and English vision reference"]
fn fixed_work_decoder_benchmark() {
    let model = Model::load_with_quantization(
        Path::new("artifacts/upstream"),
        Some(Path::new("artifacts/unlimited-ocr-q8-block32.safetensors")),
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
        for token in tokens.iter().cycle().take(512) {
            let logits = decoder.step(*token, &mut state, &mut cpu).unwrap();
            // Consume every output; the reported duration includes this checksum.
            for value in &logits.data {
                checksum = checksum.wrapping_add(u64::from(value.to_bits()));
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
                "logits_checksum": checksum, "precision": model.precision(),
                "input_policy": "recorded English tokens cycled; teacher forced; ignores EOS"
            })
        );
    }
}
