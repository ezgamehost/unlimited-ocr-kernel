use std::{hint::black_box, time::Instant};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};
fn main() {
    let model = Checkpoint::open(std::path::Path::new(
        "artifacts/upstream/model-00001-of-000001.safetensors",
    ))
    .unwrap();
    for name in [
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "lm_head.weight",
    ] {
        let weight = model.tensor(name).unwrap();
        let x = Tensor::new(vec![1, weight.shape[1]], vec![0.1; weight.shape[1]]).unwrap();
        let repeats = if weight.shape[0] > 10000 { 10 } else { 200 };
        for threads in [1, 2, 4, 8, 10] {
            let mut cpu = CpuContext::new(Backend::Auto, threads, 1 << 30).unwrap();
            for _ in 0..3 {
                black_box(ops::linear_weight(&x, weight, None, &mut cpu).unwrap());
            }
            for worker in [false, true] {
                let run = |cpu: &mut CpuContext| {
                    for _ in 0..repeats {
                        black_box(ops::linear_weight(&x, weight, None, cpu).unwrap());
                    }
                };
                let start = Instant::now();
                if worker {
                    cpu.run(run)
                } else {
                    run(&mut cpu)
                };
                println!(
                    "weight={name} threads={threads} worker={worker} us_per_call={:.2}",
                    start.elapsed().as_secs_f64() * 1e6 / repeats as f64
                );
            }
        }
    }
}
