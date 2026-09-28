mod clip;
mod sam;
mod spatial;
use crate::{
    checkpoint::Checkpoint,
    cpu::{CpuContext, ops},
    error::{OcrError, Result},
    preprocess::PreparedImage,
    tensor::Tensor,
};
pub use spatial::{window_partition, window_unpartition};
use std::collections::BTreeMap;

/// Vision weights alone are cached in FP32 for dense Accelerate operations.
/// Decoder/expert weights remain memory-mapped BF16.
pub struct Vision {
    weights: BTreeMap<String, Tensor>,
}

impl Vision {
    pub fn new(checkpoint: &Checkpoint) -> Result<Self> {
        let mut weights = BTreeMap::new();
        for name in checkpoint.names().filter(|n| {
            n.starts_with("model.sam_model.")
                || n.starts_with("model.vision_model.")
                || n.starts_with("model.projector.")
                || *n == "model.image_newline"
                || *n == "model.view_seperator"
        }) {
            let w = checkpoint.tensor(name)?;
            weights.insert(name.to_owned(), Tensor::new(w.shape.to_vec(), w.to_f32())?);
        }
        let model = Self { weights };
        if model.w("model.sam_model.patch_embed.proj.weight")?.shape != [768, 3, 16, 16] {
            return Err(OcrError::Unsupported("vision patch encoder shape".into()));
        }
        Ok(model)
    }
    fn w(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| OcrError::Format(format!("missing vision tensor {name}")))
    }
    fn linear(&self, x: &Tensor, name: &str, cpu: &mut CpuContext) -> Result<Tensor> {
        let bias = self
            .weights
            .get(&format!("{name}.bias"))
            .map(|x| x.data.as_slice());
        ops::linear(x, self.w(&format!("{name}.weight"))?, bias, cpu)
    }
    fn norm(&self, x: &Tensor, name: &str, eps: f32) -> Result<Tensor> {
        ops::layer_norm(
            x,
            &self.w(&format!("{name}.weight"))?.data,
            Some(&self.w(&format!("{name}.bias"))?.data),
            eps,
        )
    }
    fn conv(
        &self,
        x: &Tensor,
        name: &str,
        stride: usize,
        pad: usize,
        cpu: &mut CpuContext,
    ) -> Result<Tensor> {
        let bias = self
            .weights
            .get(&format!("{name}.bias"))
            .map(|x| x.data.as_slice());
        ops::conv2d(
            x,
            self.w(&format!("{name}.weight"))?,
            bias,
            stride,
            pad,
            cpu,
        )
    }
    fn project(&self, sam: &Tensor, clip: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        let count = sam.shape[1] * sam.shape[2];
        if sam.shape[3] != 1024 || clip.shape != [1, count + 1, 1024] {
            return Err(OcrError::Shape("vision fusion shapes".into()));
        }
        let mut combined = Vec::with_capacity(count * 2048);
        for p in 0..count {
            combined.extend_from_slice(&clip.data[(p + 1) * 1024..(p + 2) * 1024]);
            combined.extend_from_slice(&sam.data[p * 1024..(p + 1) * 1024]);
        }
        self.linear(
            &Tensor::new(vec![count, 2048], combined)?,
            "model.projector.layers",
            cpu,
        )
    }
    fn sequence(&self, grid: &Tensor, height: usize, width: usize) -> Result<Vec<f32>> {
        if grid.data.len() != height * width * 1280 {
            return Err(OcrError::Shape("projected image grid".into()));
        }
        let newline = &self.w("model.image_newline")?.data;
        if newline.len() != 1280 {
            return Err(OcrError::Shape("image newline width".into()));
        }
        let mut out = Vec::with_capacity(height * (width + 1) * 1280);
        for row in grid.data.chunks_exact(width * 1280) {
            out.extend_from_slice(row);
            out.extend_from_slice(newline);
        }
        Ok(out)
    }
    pub fn encode(&self, image: &PreparedImage, cpu: &mut CpuContext) -> Result<Tensor> {
        Ok(self.encode_debug(image, cpu)?.2)
    }

    /// Intermediate outputs for a single global view or local crop.
    pub fn encode_view_debug(
        &self,
        image: &Tensor,
        cpu: &mut CpuContext,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let sam = self.sam(image, cpu)?;
        let clip = self.clip(&sam, cpu)?;
        let projected = self.project(&sam, &clip, cpu)?;
        Ok((sam, clip, projected))
    }

    /// Diagnostic stage outputs support numerical audits and per-stage profiling.
    pub fn encode_debug(
        &self,
        image: &PreparedImage,
        cpu: &mut CpuContext,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let image_tokens = image.image_tokens()?;
        let (sam, clip, global) = self.encode_view_debug(&image.global, cpu)?;
        let mut sequence = Vec::new();
        let [gx, gy] = image.grid;
        if !image.crops.is_empty() {
            if image.crops.len() != gx * gy {
                return Err(OcrError::Shape("crop grid mismatch".into()));
            }
            let mut grid = vec![0.; gy * 10 * gx * 10 * 1280];
            for (index, crop) in image.crops.iter().enumerate() {
                let (_, _, local) = self.encode_view_debug(crop, cpu)?;
                if local.shape != [100, 1280] {
                    return Err(OcrError::Shape("crop embedding shape".into()));
                }
                for y in 0..10 {
                    for x in 0..10 {
                        let dest = ((index / gx * 10 + y) * gx * 10 + index % gx * 10 + x) * 1280;
                        grid[dest..dest + 1280].copy_from_slice(
                            &local.data[(y * 10 + x) * 1280..(y * 10 + x + 1) * 1280],
                        );
                    }
                }
            }
            sequence = self.sequence(
                &Tensor::new(vec![gy * 10, gx * 10, 1280], grid)?,
                gy * 10,
                gx * 10,
            )?;
        }
        sequence.extend(self.sequence(&global, 16, 16)?);
        let separator = &self.w("model.view_seperator")?.data;
        if separator.len() != 1280 {
            return Err(OcrError::Shape("image separator width".into()));
        }
        sequence.extend_from_slice(separator);
        let result = Tensor::new(vec![image_tokens, 1280], sequence)?;
        Ok((sam, clip, result))
    }
}

fn residual(mut x: Tensor, y: Tensor) -> Result<Tensor> {
    if x.shape != y.shape {
        return Err(OcrError::Shape("residual shape mismatch".into()));
    }
    for (a, b) in x.data.iter_mut().zip(y.data) {
        *a += b;
    }
    Ok(x)
}
