//! Diagnostic only: compare identical attention arithmetic at two KV strides.
use super::{Backend, CpuContext, accelerate, ops};
use rayon::prelude::*;
use std::time::Instant;

const HEADS: usize = 10;
const DIM: usize = 128;
const WIDTH: usize = HEADS * DIM;
const TOKENS: usize = 1641;

struct Layer {
    q: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    packed_keys: Vec<f32>,
    packed_values: Vec<f32>,
}

fn attention(layer: &Layer, packed: bool) -> Vec<f32> {
    let mut output = vec![0.; WIDTH];
    let mut scores = vec![0.; TOKENS * HEADS];
    output
        .par_chunks_mut(DIM)
        .zip(scores.par_chunks_mut(TOKENS))
        .enumerate()
        .for_each(|(head, (dest, scores))| {
            let (keys, values, offset, stride) = if packed {
                (
                    &layer.packed_keys,
                    &layer.packed_values,
                    head * TOKENS * DIM,
                    DIM,
                )
            } else {
                (&layer.keys, &layer.values, head * DIM, WIDTH)
            };
            accelerate::matvec_strided(
                &keys[offset..],
                &layer.q[head * DIM..(head + 1) * DIM],
                scores,
                TOKENS,
                DIM,
                stride,
                false,
            )
            .unwrap();
            for value in scores.iter_mut() {
                *value *= (DIM as f32).sqrt().recip();
            }
            ops::softmax_backend(scores, Backend::Auto).unwrap();
            accelerate::matvec_strided(&values[offset..], scores, dest, TOKENS, DIM, stride, true)
                .unwrap();
        });
    output
}

#[test]
#[ignore = "isolated attention layout timing probe; macOS Accelerate required"]
fn compare_kv_strides() {
    if !cfg!(target_os = "macos") || !accelerate::parallel_matvec_supported() {
        return;
    }
    let layers: Vec<_> = (0..12)
        .map(|layer| {
            let q = (0..WIDTH)
                .map(|i| ((i + layer * 31) as f32 * 0.017).sin())
                .collect();
            let keys: Vec<_> = (0..TOKENS * WIDTH)
                .map(|i| ((i + layer * 71) as f32 * 0.0013).cos())
                .collect();
            let values: Vec<_> = (0..TOKENS * WIDTH)
                .map(|i| ((i + layer * 97) as f32 * 0.0031).sin())
                .collect();
            let pack = |input: &[f32]| {
                let mut out = vec![0.; input.len()];
                for head in 0..HEADS {
                    for token in 0..TOKENS {
                        out[(head * TOKENS + token) * DIM..(head * TOKENS + token + 1) * DIM]
                            .copy_from_slice(
                                &input
                                    [token * WIDTH + head * DIM..token * WIDTH + (head + 1) * DIM],
                            );
                    }
                }
                out
            };
            Layer {
                q,
                packed_keys: pack(&keys),
                packed_values: pack(&values),
                keys,
                values,
            }
        })
        .collect();
    let cpu = CpuContext::new(Backend::Auto, 8, 1 << 30).unwrap();
    cpu.pool.install(|| {
        for layer in &layers {
            let a = attention(layer, false);
            let b = attention(layer, true);
            let differences = a.iter().zip(&b).filter(|(a,b)| a.to_bits() != b.to_bits()).count();
            println!("{}", serde_json::json!({"parity_differences":differences}));
            assert_eq!(differences, 0, "stride change altered attention arithmetic");
        }
        for (phase, packed) in [false, true, true, false].into_iter().enumerate() {
            for repetition in 0..4 {
                let start = Instant::now();
                let mut checksum = 0u64;
                for _ in 0..128 {
                    for layer in &layers {
                        let out = attention(layer, packed);
                        checksum = checksum.wrapping_add(u64::from(out[0].to_bits()));
                        std::hint::black_box(out);
                    }
                }
                println!("{}", serde_json::json!({"phase":phase,"packed":packed,"repetition":repetition,
                    "warmup":repetition==0,"seconds":start.elapsed().as_secs_f64(),"checksum":checksum,
                    "layers":12,"steps":128,"tokens":TOKENS,"heads":HEADS,"dim":DIM,"threads":8}));
            }
        }
    });
}
