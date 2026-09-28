use std::{hint::black_box, time::Instant};
use uocr::{
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};
fn main() {
    let (heads, dim) = (10, 128);
    let width = heads * dim;
    let q = Tensor::new(vec![1, width], vec![0.1; width]).unwrap();
    let qh = Tensor::new(vec![heads, 1, dim], q.data.clone()).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 10, 1 << 30).unwrap();
    for tokens in [405, 1641, 8192] {
        let k: Vec<f32> = (0..tokens * width)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let v = k.clone();
        let split = |x: &[f32]| {
            let mut data = vec![0.; x.len()];
            for head in 0..heads {
                for token in 0..tokens {
                    data[(head * tokens + token) * dim..(head * tokens + token + 1) * dim]
                        .copy_from_slice(
                            &x[token * width + head * dim..token * width + (head + 1) * dim],
                        );
                }
            }
            Tensor::new(vec![heads, tokens, dim], data).unwrap()
        };
        for _ in 0..3 {
            for direct in [false, true] {
                let start = Instant::now();
                for _ in 0..20 {
                    let result = if direct {
                        ops::attention_decode(&q, &k, &v, heads, &mut cpu)
                    } else {
                        ops::attention(&qh, &split(&k), &split(&v), false, &mut cpu)
                    }
                    .unwrap();
                    black_box(result);
                }
                println!(
                    "tokens={tokens} direct={direct} ms_per_layer={:.4}",
                    start.elapsed().as_secs_f64() * 1000. / 20.
                );
            }
        }
    }
}
