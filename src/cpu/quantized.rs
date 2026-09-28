//! Blockwise symmetric INT8 weights with FP32 activations and accumulation.
use super::{Backend, CpuContext, ops};
use crate::{
    error::{OcrError, Result},
    tensor::Tensor,
};
use rayon::prelude::*;

pub const BLOCK: usize = 32;

pub struct Int8Matrix {
    rows: usize,
    cols: usize,
    weights: Vec<i8>,
    scales: Vec<f32>,
}
impl Int8Matrix {
    pub fn from_parts(
        rows: usize,
        cols: usize,
        weights: Vec<i8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        if rows == 0
            || cols == 0
            || rows.checked_mul(cols) != Some(weights.len())
            || rows.checked_mul(cols.div_ceil(BLOCK)) != Some(scales.len())
        {
            return Err(OcrError::Shape("INT8 matrix storage".into()));
        }
        if scales.iter().any(|s| !s.is_finite() || *s < 0.) {
            return Err(OcrError::Numerical(
                "INT8 scales must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            rows,
            cols,
            weights,
            scales,
        })
    }
    pub fn quantize(matrix: &Tensor) -> Result<Self> {
        matrix.validate()?;
        if matrix.shape.len() != 2 || matrix.shape.contains(&0) {
            return Err(OcrError::Shape(
                "quantization expects a nonempty matrix".into(),
            ));
        }
        if matrix.data.iter().any(|v| !v.is_finite()) {
            return Err(OcrError::Numerical("nonfinite quantization input".into()));
        }
        let (rows, cols) = (matrix.shape[0], matrix.shape[1]);
        let mut weights = Vec::with_capacity(matrix.data.len());
        let mut scales = Vec::with_capacity(rows * cols.div_ceil(BLOCK));
        for row in matrix.data.chunks_exact(cols) {
            for block in row.chunks(BLOCK) {
                let max = block.iter().map(|v| v.abs()).fold(0., f32::max);
                let scale = max / 127.;
                if max > 0. && scale == 0. {
                    return Err(OcrError::Numerical("INT8 scale underflow".into()));
                }
                scales.push(scale);
                for &value in block {
                    weights.push(if scale == 0. {
                        0
                    } else {
                        (value / scale).round_ties_even().clamp(-127., 127.) as i8
                    });
                }
            }
        }
        Self::from_parts(rows, cols, weights, scales)
    }
    pub fn dequantize(&self) -> Result<Tensor> {
        let blocks = self.cols.div_ceil(BLOCK);
        let data = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, &w)| w as f32 * self.scales[i / self.cols * blocks + (i % self.cols) / BLOCK])
            .collect();
        Tensor::new(vec![self.rows, self.cols], data)
    }
    pub fn storage_bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
    }
    pub fn linear(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        x.validate()?;
        if x.shape.last() != Some(&self.cols) {
            return Err(OcrError::Shape("INT8 linear input width".into()));
        }
        let batch = x.data.len() / self.cols;
        let count = batch
            .checked_mul(self.rows)
            .ok_or_else(|| OcrError::Resource("INT8 output overflow".into()))?;
        cpu.check_elements(count)?;
        if batch != 1 {
            cpu.check_elements(
                self.weights
                    .len()
                    .checked_add(count)
                    .ok_or_else(|| OcrError::Resource("INT8 scratch overflow".into()))?,
            )?;
            return ops::linear(x, &self.dequantize()?, None, cpu);
        }
        let blocks = self.cols.div_ceil(BLOCK);
        let dot = |row: usize| {
            let w = &self.weights[row * self.cols..(row + 1) * self.cols];
            let scales = &self.scales[row * blocks..(row + 1) * blocks];
            #[cfg(target_arch = "aarch64")]
            if cpu.backend != Backend::Scalar {
                return super::neon::dot_int8(w, &x.data, scales);
            }
            w.iter()
                .zip(&x.data)
                .enumerate()
                .map(|(i, (&w, &v))| w as f32 * scales[i / BLOCK] * v)
                .sum()
        };
        let mut data = vec![0.; self.rows];
        if cpu.backend == Backend::Scalar || cpu.threads() == 1 || self.weights.len() <= 128 * 1024
        {
            for (row, y) in data.iter_mut().enumerate() {
                *y = dot(row);
            }
        } else {
            cpu.pool.install(|| {
                data.par_iter_mut()
                    .with_min_len(self.rows.div_ceil(cpu.threads()))
                    .enumerate()
                    .for_each(|(row, y)| *y = dot(row))
            });
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = self.rows;
        Tensor::new(shape, data)
    }
}

impl Int8Matrix {
    /// Experimental single-token W8A8 path. Rounds each activation block to
    /// INT8; unlike `linear`, this adds activation quantization error.
    pub fn linear_dynamic_int8(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        x.validate()?;
        if x.shape.last() != Some(&self.cols) || x.data.len() != self.cols {
            return Err(OcrError::Shape(
                "dynamic INT8 requires one input row".into(),
            ));
        }
        let scratch = self
            .cols
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.rows))
            .and_then(|n| n.checked_add(self.cols.div_ceil(BLOCK)))
            .ok_or_else(|| OcrError::Resource("dynamic INT8 scratch overflow".into()))?;
        cpu.check_elements(scratch)?;
        let activation = Self::quantize(&Tensor::new(vec![1, self.cols], x.data.clone())?)?;
        let blocks = self.cols.div_ceil(BLOCK);
        #[cfg(target_arch = "aarch64")]
        let dotprod =
            cpu.backend != Backend::Scalar && std::arch::is_aarch64_feature_detected!("dotprod");
        let dot = |row: usize| {
            let weights = &self.weights[row * self.cols..(row + 1) * self.cols];
            let scales = &self.scales[row * blocks..(row + 1) * blocks];
            #[cfg(target_arch = "aarch64")]
            if dotprod {
                // SAFETY: runtime detection proves the dotprod ISA is available;
                // validated matrix/activation storage has equal lengths/scales.
                return unsafe {
                    super::neon::dot_int8_pair(
                        weights,
                        &activation.weights,
                        scales,
                        &activation.scales,
                    )
                };
            }
            weights
                .chunks(BLOCK)
                .zip(activation.weights.chunks(BLOCK))
                .enumerate()
                .map(|(b, (w, x))| {
                    let sum: i32 = w.iter().zip(x).map(|(&w, &x)| w as i32 * x as i32).sum();
                    sum as f32 * scales[b] * activation.scales[b]
                })
                .sum()
        };
        let mut data = vec![0.; self.rows];
        if cpu.backend == Backend::Scalar || cpu.threads() == 1 || self.weights.len() <= 128 * 1024
        {
            for (row, y) in data.iter_mut().enumerate() {
                *y = dot(row);
            }
        } else {
            cpu.pool.install(|| {
                data.par_iter_mut()
                    .with_min_len(self.rows.div_ceil(cpu.threads()))
                    .enumerate()
                    .for_each(|(row, y)| *y = dot(row))
            });
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = self.rows;
        Tensor::new(shape, data)
    }
}
