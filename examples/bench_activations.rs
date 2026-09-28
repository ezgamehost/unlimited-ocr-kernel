use std::{hint::black_box, time::Instant};
use uocr::cpu::{Backend, CpuContext, ops};

fn main() {
    let input: Vec<f32> = (0..4096 * 256)
        .map(|i| (i as f32 * 0.13).sin() * 5.)
        .collect();
    let mut data = input.clone();
    for activation in ["gelu", "quick_gelu"] {
        for _ in 0..4 {
            for backend in [Backend::Scalar, Backend::Auto] {
                let cpu = CpuContext::new(backend, 8, 1 << 20).unwrap();
                let start = Instant::now();
                for _ in 0..16 {
                    data.copy_from_slice(&input);
                    if activation == "gelu" {
                        ops::gelu_parallel(black_box(&mut data), &cpu);
                    } else {
                        ops::quick_gelu_parallel(black_box(&mut data), &cpu);
                    }
                    black_box(&data);
                }
                println!(
                    "{activation} backend={backend:?} seconds={:.6}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
