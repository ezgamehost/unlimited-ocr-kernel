use std::{hint::black_box, path::Path, time::Instant};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext, ops, quantized::Int8Matrix},
    tensor::Tensor,
};
fn main() {
    let weights = Checkpoint::open(Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    for name in [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "lm_head.weight",
    ] {
        let bf16 = weights.tensor(name).unwrap();
        let source = Tensor::new(bf16.shape.to_vec(), bf16.to_f32()).unwrap();
        let q = Int8Matrix::quantize(&source).unwrap();
        drop(source);
        let x = Tensor::new(
            vec![1, bf16.shape[1]],
            (0..bf16.shape[1]).map(|i| (i as f32 * 0.3).sin()).collect(),
        )
        .unwrap();
        println!(
            "weight={name} bf16_bytes={} int8_bytes={}",
            bf16.bytes.len(),
            q.storage_bytes()
        );
        let repeats = if bf16.shape[0] > 10000 { 20 } else { 500 };
        for mode in [0, 1, 2, 2, 1, 0] {
            let seconds = cpu.run(|cpu| {
                let start = Instant::now();
                for _ in 0..repeats {
                    let y = match mode {
                        1 => q.linear(&x, cpu),
                        2 => q.linear_dynamic_int8(&x, cpu),
                        _ => ops::linear_weight(&x, bf16, None, cpu),
                    }
                    .unwrap();
                    black_box(y);
                }
                start.elapsed().as_secs_f64()
            });
            println!(
                "mode={mode} us_per_call={:.3}",
                seconds * 1e6 / repeats as f64
            );
        }
    }
}
