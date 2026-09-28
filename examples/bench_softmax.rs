use std::{hint::black_box, time::Instant};
use uocr::cpu::{Backend, ops};

fn main() {
    for width in [196, 4096] {
        let input: Vec<f32> = (0..width).map(|i| (i as f32 * 0.37).sin() * 12.).collect();
        let mut row = input.clone();
        for _ in 0..3 {
            for backend in [Backend::Scalar, Backend::Auto] {
                let start = Instant::now();
                for _ in 0..(16_000_000 / width) {
                    row.copy_from_slice(&input);
                    ops::softmax_backend(black_box(&mut row), backend).unwrap();
                    black_box(&row);
                }
                println!(
                    "width={width} backend={backend:?} seconds={:.6}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
