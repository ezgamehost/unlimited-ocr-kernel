use super::{Vision, residual, spatial::interpolate_grid};
use crate::{
    cpu::{CpuContext, ops},
    error::{OcrError, Result},
    tensor::Tensor,
};

impl Vision {
    pub(super) fn clip(&self, sam: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        if sam.shape.len() != 4 || sam.shape[0] != 1 || sam.shape[3] != 1024 {
            return Err(OcrError::Shape("CLIP SAM features".into()));
        }
        let (h, w) = (sam.shape[1], sam.shape[2]);
        let tokens = h * w + 1;
        let width = 1024;
        let position = self.w("model.vision_model.embeddings.position_embedding.weight")?;
        let grid = Tensor::new(vec![1, 16, 16, width], position.data[width..].to_vec())?;
        let grid = interpolate_grid(&grid, h, w)?;
        let mut data = self
            .w("model.vision_model.embeddings.class_embedding")?
            .data
            .clone();
        data.extend_from_slice(&sam.data);
        for (value, pos) in data[..width].iter_mut().zip(&position.data[..width]) {
            *value += pos;
        }
        for (value, pos) in data[width..].iter_mut().zip(grid.data) {
            *value += pos;
        }
        let mut x = self.norm(
            &Tensor::new(vec![1, tokens, width], data)?,
            "model.vision_model.pre_layrnorm",
            1e-5,
        )?;
        for layer in 0..24 {
            let prefix = format!("model.vision_model.transformer.layers.{layer}");
            let norm = self.norm(&x, &format!("{prefix}.layer_norm1"), 1e-5)?;
            let qkv = self.linear(&norm, &format!("{prefix}.self_attn.qkv_proj"), cpu)?;
            let mut q = vec![0.; tokens * width];
            let mut k = q.clone();
            let mut v = q.clone();
            for head in 0..16 {
                for token in 0..tokens {
                    let dst = (head * tokens + token) * 64;
                    let src = token * 3 * width + head * 64;
                    q[dst..dst + 64].copy_from_slice(&qkv.data[src..src + 64]);
                    k[dst..dst + 64].copy_from_slice(&qkv.data[src + width..src + width + 64]);
                    v[dst..dst + 64]
                        .copy_from_slice(&qkv.data[src + 2 * width..src + 2 * width + 64]);
                }
            }
            let attended = ops::attention(
                &Tensor::new(vec![16, tokens, 64], q)?,
                &Tensor::new(vec![16, tokens, 64], k)?,
                &Tensor::new(vec![16, tokens, 64], v)?,
                false,
                cpu,
            )?;
            let mut joined = vec![0.; tokens * width];
            for head in 0..16 {
                for token in 0..tokens {
                    joined[token * width + head * 64..token * width + (head + 1) * 64]
                        .copy_from_slice(
                            &attended.data
                                [(head * tokens + token) * 64..(head * tokens + token + 1) * 64],
                        );
                }
            }
            let attention = self.linear(
                &Tensor::new(vec![1, tokens, width], joined)?,
                &format!("{prefix}.self_attn.out_proj"),
                cpu,
            )?;
            x = residual(x, attention)?;
            let norm = self.norm(&x, &format!("{prefix}.layer_norm2"), 1e-5)?;
            let mut up = self.linear(&norm, &format!("{prefix}.mlp.fc1"), cpu)?;
            ops::quick_gelu_parallel(&mut up.data, cpu);
            x = residual(x, self.linear(&up, &format!("{prefix}.mlp.fc2"), cpu)?)?;
        }
        Ok(x)
    }
}
