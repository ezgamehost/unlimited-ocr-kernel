use super::{
    Vision, residual,
    spatial::{interpolate_grid, interpolate_relative, window_partition, window_unpartition},
};
use crate::{
    cpu::{CpuContext, ops},
    error::{OcrError, Result},
    tensor::Tensor,
};
use rayon::prelude::*;

fn add_relative_bias(
    scores: &mut [f32],
    head: usize,
    start: usize,
    (h, w): (usize, usize),
    (bh, bw): (&[f32], &[f32]),
    pool: &rayon::ThreadPool,
    threads: usize,
) {
    let tokens = h * w;
    let add_row = |(i, row): (usize, &mut [f32])| {
        let query = start + i;
        let (qy, qx) = (query / w, query % w);
        let hbase = (head * tokens + query) * (2 * h - 1) + qy;
        let wbase = (head * tokens + query) * (2 * w - 1) + qx;
        let horizontal = &bw[wbase..wbase + w];
        for (key_row, &vertical) in row
            .chunks_exact_mut(w)
            .zip(bh[hbase..hbase + h].iter().rev())
        {
            for (value, &horizontal) in key_row.iter_mut().zip(horizontal.iter().rev()) {
                *value += vertical + horizontal;
            }
        }
    };
    if threads > 1 && scores.len() >= 65536 {
        let rows_per_task = (scores.len() / tokens).div_ceil(threads);
        pool.install(|| {
            scores
                .par_chunks_mut(tokens)
                .with_min_len(rows_per_task)
                .enumerate()
                .for_each(add_row)
        });
    } else {
        scores.chunks_mut(tokens).enumerate().for_each(add_row);
    }
}

impl Vision {
    pub(super) fn sam(&self, image: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        if image.shape.len() != 4
            || image.shape[0] != 1
            || image.shape[1] != image.shape[2]
            || ![640, 1024].contains(&image.shape[1])
            || image.shape[3] != 3
        {
            return Err(OcrError::Shape(
                "SAM expects a 640 or 1024 RGB square".into(),
            ));
        }
        let mut x = self.conv(image, "model.sam_model.patch_embed.proj", 16, 0, cpu)?;
        let position =
            interpolate_grid(self.w("model.sam_model.pos_embed")?, x.shape[1], x.shape[2])?;
        x = residual(x, position)?;
        for layer in 0..12 {
            let prefix = format!("model.sam_model.blocks.{layer}");
            let norm = self.norm(&x, &format!("{prefix}.norm1"), 1e-6)?;
            let windowed = ![2, 5, 8, 11].contains(&layer);
            let normalized = if windowed {
                window_partition(&norm, 14)?
            } else {
                norm
            };
            let attended = self.sam_attention(&normalized, &format!("{prefix}.attn"), cpu)?;
            let attended = if windowed {
                window_unpartition(&attended, x.shape[1], x.shape[2], 14)?
            } else {
                attended
            };
            x = residual(x, attended)?;
            let norm = self.norm(&x, &format!("{prefix}.norm2"), 1e-6)?;
            let mut up = self.linear(&norm, &format!("{prefix}.mlp.lin1"), cpu)?;
            ops::gelu_parallel(&mut up.data, cpu);
            x = residual(x, self.linear(&up, &format!("{prefix}.mlp.lin2"), cpu)?)?;
        }
        x = self.conv(&x, "model.sam_model.neck.0", 1, 0, cpu)?;
        x = self.norm(&x, "model.sam_model.neck.1", 1e-6)?;
        x = self.conv(&x, "model.sam_model.neck.2", 1, 1, cpu)?;
        x = self.norm(&x, "model.sam_model.neck.3", 1e-6)?;
        x = self.conv(&x, "model.sam_model.net_2", 2, 1, cpu)?;
        self.conv(&x, "model.sam_model.net_3", 2, 1, cpu)
    }

    fn sam_attention(&self, x: &Tensor, prefix: &str, cpu: &mut CpuContext) -> Result<Tensor> {
        let (batch, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let heads = 12;
        let dim = c / heads;
        let tokens = h * w;
        let qkv = self.linear(x, &format!("{prefix}.qkv"), cpu)?;
        let mut q = vec![0.; batch * heads * tokens * dim];
        let mut k = q.clone();
        let mut v = q.clone();
        for b in 0..batch {
            for head in 0..heads {
                for token in 0..tokens {
                    let dst = ((b * heads + head) * tokens + token) * dim;
                    let src = (b * tokens + token) * 3 * c + head * dim;
                    q[dst..dst + dim].copy_from_slice(&qkv.data[src..src + dim]);
                    k[dst..dst + dim].copy_from_slice(&qkv.data[src + c..src + c + dim]);
                    v[dst..dst + dim].copy_from_slice(&qkv.data[src + 2 * c..src + 2 * c + dim]);
                }
            }
        }
        let q = Tensor::new(vec![batch * heads, tokens, dim], q)?;
        let k = Tensor::new(q.shape.clone(), k)?;
        let v = Tensor::new(q.shape.clone(), v)?;
        let rh = interpolate_relative(self.w(&format!("{prefix}.rel_pos_h"))?, 2 * h - 1)?;
        let rw = interpolate_relative(self.w(&format!("{prefix}.rel_pos_w"))?, 2 * w - 1)?;
        let bh = ops::linear(&q, &rh, None, cpu)?;
        let bw = ops::linear(&q, &rw, None, cpu)?;
        let bias_pool = std::sync::Arc::clone(&cpu.pool);
        let workers = if cpu.backend == crate::cpu::Backend::Scalar {
            1
        } else {
            cpu.threads()
        };
        let attended = ops::attention_bias(&q, &k, &v, false, cpu, |head, start, _, scores| {
            add_relative_bias(
                scores,
                head,
                start,
                (h, w),
                (&bh.data, &bw.data),
                &bias_pool,
                workers,
            );
        })?;
        let mut joined = vec![0.; batch * tokens * c];
        for b in 0..batch {
            for head in 0..heads {
                for token in 0..tokens {
                    let src = ((b * heads + head) * tokens + token) * dim;
                    let dst = (b * tokens + token) * c + head * dim;
                    joined[dst..dst + dim].copy_from_slice(&attended.data[src..src + dim]);
                }
            }
        }
        self.linear(
            &Tensor::new(x.shape.clone(), joined)?,
            &format!("{prefix}.proj"),
            cpu,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_bias_rows_preserve_indexed_formula() {
        for (h, w, start, count) in [(3, 5, 2, 9), (40, 40, 37, 127), (64, 64, 55, 128)] {
            let tokens = h * w;
            let bh: Vec<f32> = (0..2 * tokens * (2 * h - 1))
                .map(|i| (i as f32 * 0.03).sin())
                .collect();
            let bw: Vec<f32> = (0..2 * tokens * (2 * w - 1))
                .map(|i| (i as f32 * 0.13).cos())
                .collect();
            let mut expected: Vec<f32> = (0..count * tokens)
                .map(|i| (i as f32 * 0.23).sin())
                .collect();
            let original = expected.clone();
            for i in 0..count {
                let query = start + i;
                for ky in 0..h {
                    for kx in 0..w {
                        expected[i * tokens + ky * w + kx] += bh
                            [(tokens + query) * (2 * h - 1) + query / w + h - 1 - ky]
                            + bw[(tokens + query) * (2 * w - 1) + query % w + w - 1 - kx];
                    }
                }
            }
            for threads in [1, 8] {
                let cpu = CpuContext::new(crate::cpu::Backend::Auto, threads, 1 << 24).unwrap();
                let mut actual = original.clone();
                add_relative_bias(
                    &mut actual,
                    1,
                    start,
                    (h, w),
                    (&bh, &bw),
                    &cpu.pool,
                    threads,
                );
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }
}
