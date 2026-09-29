//! Block32 signed INT4 weights: low sixteen codes, then high sixteen codes.
use super::{
    Backend, CpuContext, ops,
    quantized::{BLOCK, QuantizedActivation},
};
use crate::{
    error::{OcrError, Result},
    tensor::Tensor,
};
use rayon::prelude::*;

pub struct Int4Matrix {
    rows: usize,
    cols: usize,
    blocks: usize,
    weights: Vec<u8>,
    scales: Vec<f32>,
}

fn signed(nibble: u8) -> i8 {
    ((nibble as i8) << 4) >> 4
}

impl Int4Matrix {
    pub fn from_parts(
        rows: usize,
        cols: usize,
        weights: Vec<u8>,
        scales: Vec<f32>,
    ) -> Result<Self> {
        let blocks = cols.div_ceil(BLOCK);
        if rows == 0
            || cols == 0
            || rows
                .checked_mul(cols)
                .and_then(|n| n.checked_mul(4))
                .is_none()
            || rows.checked_mul(blocks).and_then(|n| n.checked_mul(16)) != Some(weights.len())
            || rows.checked_mul(blocks) != Some(scales.len())
        {
            return Err(OcrError::Shape("INT4 matrix storage".into()));
        }
        if scales.iter().any(|s| !s.is_finite() || *s < 0.) {
            return Err(OcrError::Numerical(
                "INT4 scales must be finite and nonnegative".into(),
            ));
        }
        if weights.iter().any(|&v| v & 15 == 8 || v >> 4 == 8) {
            return Err(OcrError::Format("INT4 code -8 is forbidden".into()));
        }
        let matrix = Self {
            rows,
            cols,
            blocks,
            weights,
            scales,
        };
        if !cols.is_multiple_of(BLOCK) {
            for row in 0..rows {
                for tail in cols % BLOCK..BLOCK {
                    let byte = matrix.weights[(row * blocks + blocks - 1) * 16 + tail % 16];
                    if (byte >> (4 * (tail / 16))) & 15 != 0 {
                        return Err(OcrError::Format("INT4 nonzero tail padding".into()));
                    }
                }
            }
        }
        Ok(matrix)
    }

    fn code(&self, row: usize, col: usize) -> i8 {
        let byte = self.weights[(row * self.blocks + col / BLOCK) * 16 + col % 16];
        signed(byte >> (4 * ((col % BLOCK) / 16)))
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        let mut data = vec![0.; self.rows * self.cols];
        for ((weights, scales), output) in self
            .weights
            .chunks_exact(self.blocks * 16)
            .zip(self.scales.chunks_exact(self.blocks))
            .zip(data.chunks_exact_mut(self.cols))
        {
            for ((packed, &scale), output) in weights
                .as_chunks::<16>()
                .0
                .iter()
                .zip(scales)
                .zip(output.chunks_mut(BLOCK))
            {
                let split = output.len().min(16);
                let (low, high) = output.split_at_mut(split);
                for (value, &byte) in low.iter_mut().zip(packed) {
                    *value = signed(byte) as f32 * scale;
                }
                for (value, &byte) in high.iter_mut().zip(packed) {
                    *value = signed(byte >> 4) as f32 * scale;
                }
            }
        }
        Tensor::new(vec![self.rows, self.cols], data)
    }

    pub fn storage_bytes(&self) -> usize {
        self.weights.len() + self.scales.len() * 4
    }

    fn output(
        &self,
        x: &Tensor,
        cpu: &CpuContext,
        dot: impl Fn(usize) -> f32 + Sync,
    ) -> Result<Tensor> {
        let mut data = vec![0.; self.rows];
        if cpu.backend == Backend::Scalar
            || cpu.threads() == 1
            || self.rows * self.cols <= 128 * 1024
        {
            for (row, value) in data.iter_mut().enumerate() {
                *value = dot(row);
            }
        } else {
            cpu.pool.install(|| {
                data.par_iter_mut()
                    .with_min_len(self.rows.div_ceil(cpu.threads()))
                    .enumerate()
                    .for_each(|(row, value)| *value = dot(row))
            });
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = self.rows;
        Tensor::new(shape, data)
    }

    pub fn linear(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        x.validate()?;
        if x.shape.last() != Some(&self.cols) {
            return Err(OcrError::Shape("INT4 linear input width".into()));
        }
        let batch = x.data.len() / self.cols;
        let count = batch
            .checked_mul(self.rows)
            .ok_or_else(|| OcrError::Resource("INT4 output overflow".into()))?;
        cpu.check_elements(count)?;
        if batch != 1 {
            cpu.check_elements(
                (self.rows * self.cols)
                    .checked_add(count)
                    .ok_or_else(|| OcrError::Resource("INT4 scratch overflow".into()))?,
            )?;
            return ops::linear(x, &self.dequantize()?, None, cpu);
        }
        self.output(x, cpu, |row| {
            (0..self.cols)
                .map(|col| {
                    self.code(row, col) as f32
                        * self.scales[row * self.blocks + col / BLOCK]
                        * x.data[col]
                })
                .sum()
        })
    }

    pub fn linear_dynamic_int8(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        x.validate()?;
        if x.shape.last() != Some(&self.cols) || x.data.len() != self.cols {
            return Err(OcrError::Shape(
                "dynamic INT4 requires one input row".into(),
            ));
        }
        let scratch = self
            .cols
            .checked_add(self.rows)
            .and_then(|n| n.checked_add(self.blocks))
            .ok_or_else(|| OcrError::Resource("dynamic INT4 scratch overflow".into()))?;
        cpu.check_elements(scratch)?;
        let activation = QuantizedActivation::new(x)?;
        self.linear_prepared(&activation, cpu)
    }

    /// Apply packed weights to an already quantized activation row.
    pub fn linear_prepared(
        &self,
        prepared: &QuantizedActivation,
        cpu: &mut CpuContext,
    ) -> Result<Tensor> {
        if prepared.width() != self.cols {
            return Err(OcrError::Shape("prepared INT4 activation width".into()));
        }
        let scratch = self
            .cols
            .checked_add(self.rows)
            .and_then(|n| n.checked_add(self.blocks))
            .ok_or_else(|| OcrError::Resource("dynamic INT4 scratch overflow".into()))?;
        cpu.check_elements(scratch)?;
        let (codes, scales) = prepared.matrix.parts();
        // Match the simulation backend's accumulation identity, including
        // signed zero: Iterator::sum uses -0, while SDOT starts with +0.
        let initial = -0.0f32;
        #[cfg(target_arch = "aarch64")]
        let initial = if cpu.backend != Backend::Scalar
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            0.0
        } else {
            initial
        };
        let dot = |row| {
            codes
                .chunks(BLOCK)
                .enumerate()
                .map(|(block, values)| {
                    let sum: i32 = values
                        .iter()
                        .enumerate()
                        .map(|(offset, &value)| {
                            self.code(row, block * BLOCK + offset) as i32 * value as i32
                        })
                        .sum();
                    sum as f32 * self.scales[row * self.blocks + block] * scales[block]
                })
                .fold(initial, |total, value| total + value)
        };
        let mut data = vec![0.; self.rows];
        let compute = |(group, output): (usize, &mut [f32])| {
            let first = group * 4;
            #[cfg(target_arch = "aarch64")]
            if cpu.backend != Backend::Scalar
                && std::arch::is_aarch64_feature_detected!("dotprod")
                && self.cols.is_multiple_of(BLOCK)
                && output.len() == 4
            {
                // SAFETY: dotprod detected; validated four rows of complete
                // block32 packed weights and matching activation/scales.
                let values = unsafe {
                    super::neon::dot_int4_four_rows(
                        &self.weights[first * self.blocks * 16..(first + 4) * self.blocks * 16],
                        codes,
                        &self.scales[first * self.blocks..(first + 4) * self.blocks],
                        scales,
                    )
                };
                output.copy_from_slice(&values);
                return;
            }
            for (offset, value) in output.iter_mut().enumerate() {
                *value = dot(first + offset);
            }
        };
        if cpu.backend == Backend::Scalar
            || cpu.threads() == 1
            || self.rows * self.cols <= 128 * 1024
        {
            data.chunks_mut(4).enumerate().for_each(compute);
        } else {
            cpu.pool.install(|| {
                data.par_chunks_mut(4)
                    .with_min_len(self.rows.div_ceil(4).div_ceil(cpu.threads()))
                    .enumerate()
                    .for_each(compute)
            });
        }
        let mut shape = vec![1; prepared.rank];
        *shape.last_mut().unwrap() = self.rows;
        Tensor::new(shape, data)
    }
}
