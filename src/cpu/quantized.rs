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
    pub(super) fn parts(&self) -> (&[i8], &[f32]) {
        (&self.weights, &self.scales)
    }
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
                #[cfg(target_arch = "aarch64")]
                if block.len() == BLOCK {
                    let (codes, scale) = super::neon::quantize_block32(block)
                        .ok_or_else(|| OcrError::Numerical("INT8 scale underflow".into()))?;
                    weights.extend_from_slice(&codes);
                    scales.push(scale);
                    continue;
                }
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
        let mut data = vec![0.; self.weights.len()];
        for ((weights, scales), output) in self
            .weights
            .chunks_exact(self.cols)
            .zip(self.scales.chunks_exact(blocks))
            .zip(data.chunks_exact_mut(self.cols))
        {
            for ((weights, &scale), output) in weights
                .chunks(BLOCK)
                .zip(scales)
                .zip(output.chunks_mut(BLOCK))
            {
                for (dest, &weight) in output.iter_mut().zip(weights) {
                    *dest = weight as f32 * scale;
                }
            }
        }
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
        let compute_group = |(group, output): (usize, &mut [f32])| {
            let first = group * 4;
            #[cfg(target_arch = "aarch64")]
            if dotprod && self.cols.is_multiple_of(BLOCK) && output.len() == 4 {
                // SAFETY: feature detection above; four complete validated rows
                // with columns divisible by the kernel's 32-element block.
                let result = unsafe {
                    super::neon::dot_int8_four_rows(
                        &self.weights[first * self.cols..(first + 4) * self.cols],
                        &activation.weights,
                        &self.scales[first * blocks..(first + 4) * blocks],
                        &activation.scales,
                    )
                };
                output.copy_from_slice(&result);
                return;
            }
            for (offset, value) in output.iter_mut().enumerate() {
                *value = dot(first + offset);
            }
        };
        if cpu.backend == Backend::Scalar || cpu.threads() == 1 || self.weights.len() <= 128 * 1024
        {
            data.chunks_mut(4).enumerate().for_each(compute_group);
        } else {
            cpu.pool.install(|| {
                data.par_chunks_mut(4)
                    .with_min_len(self.rows.div_ceil(4).div_ceil(cpu.threads()))
                    .enumerate()
                    .for_each(compute_group)
            });
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = self.rows;
        Tensor::new(shape, data)
    }
}

const FORMAT: &str = "uocr-q8-block32-v1";
const Q4_FORMAT: &str = "uocr-q4-block32-v1";
const HYBRID_FORMAT: &str = "uocr-q4-q8head-block32-v1";

#[derive(Clone, Copy, Debug, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMode {
    WeightsOnly,
    DynamicActivations,
}

#[derive(serde::Deserialize)]
struct Manifest {
    schema_version: u32,
    format: String,
    block_size: usize,
    source_sha256: String,
    output_sha256: String,
    matrix_count: usize,
    bytes: usize,
    weight_bits: Option<usize>,
    storage_bits: Option<usize>,
    body_bits: Option<usize>,
    head_bits: Option<usize>,
}

/// Validated, immutable owned matrices. Load-time copying also touches pages
/// before timed inference; its cost belongs in model load time.
pub enum QuantizedMatrix {
    Int8(Int8Matrix),
    Int4(super::int4::Int4Matrix),
}
impl QuantizedMatrix {
    pub fn linear(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        match self {
            Self::Int8(m) => m.linear(x, cpu),
            Self::Int4(m) => m.linear(x, cpu),
        }
    }
    pub fn linear_dynamic_int8(&self, x: &Tensor, cpu: &mut CpuContext) -> Result<Tensor> {
        match self {
            Self::Int8(m) => m.linear_dynamic_int8(x, cpu),
            Self::Int4(m) => m.linear_dynamic_int8(x, cpu),
        }
    }
}
pub struct QuantizedWeights {
    matrices: std::collections::BTreeMap<String, QuantizedMatrix>,
    packed_int4: bool,
    q8_head: bool,
}

fn eligible(name: &str) -> bool {
    if name == "lm_head.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((layer, rest)) = rest.split_once('.') else {
        return false;
    };
    if layer.parse::<usize>().is_err() {
        return false;
    }
    if let Some(part) = rest.strip_prefix("self_attn.") {
        return [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
        ]
        .contains(&part);
    }
    let Some(mut part) = rest.strip_prefix("mlp.") else {
        return false;
    };
    if let Some(rest) = part.strip_prefix("shared_experts.") {
        part = rest;
    } else if let Some(rest) = part.strip_prefix("experts.") {
        let Some((expert, rest)) = rest.split_once('.') else {
            return false;
        };
        if expert.parse::<usize>().is_err() {
            return false;
        }
        part = rest;
    }
    ["gate_proj.weight", "up_proj.weight", "down_proj.weight"].contains(&part)
}

impl QuantizedWeights {
    pub fn load(path: &std::path::Path, source: &crate::checkpoint::Checkpoint) -> Result<Self> {
        use crate::checkpoint::{Checkpoint, DType};
        let manifest: Manifest =
            serde_json::from_slice(&std::fs::read(path.with_extension("json"))?)?;
        let checkpoint = Checkpoint::open(path)?;
        let metadata = checkpoint.metadata();
        let q8_head = manifest.format == HYBRID_FORMAT;
        let packed_int4 = manifest.format == Q4_FORMAT || q8_head;
        if manifest.schema_version != 1
            || (!packed_int4 && manifest.format != FORMAT)
            || manifest.block_size != BLOCK
            || metadata.get("format") != Some(&manifest.format)
            || metadata.get("block_size").map(String::as_str) != Some("32")
            || metadata.get("source_sha256") != Some(&manifest.source_sha256)
            || checkpoint.file_bytes() != manifest.bytes
            || (packed_int4
                && !q8_head
                && (manifest.weight_bits != Some(4)
                    || manifest.storage_bits != Some(4)
                    || metadata.get("weight_bits").map(String::as_str) != Some("4")
                    || metadata.get("storage_bits").map(String::as_str) != Some("4")))
            || (q8_head
                && (manifest.body_bits != Some(4)
                    || manifest.head_bits != Some(8)
                    || metadata.get("body_bits").map(String::as_str) != Some("4")
                    || metadata.get("head_bits").map(String::as_str) != Some("8")))
        {
            return Err(OcrError::Format(
                "quantized overlay version or metadata mismatch".into(),
            ));
        }
        // Validate the large source first so the overlay is the most recently
        // read file when its matrices are copied below.
        if source.sha256()? != manifest.source_sha256
            || checkpoint.sha256()? != manifest.output_sha256
        {
            return Err(OcrError::Format(
                "quantized checksum or source model mismatch".into(),
            ));
        }
        let expected: Vec<_> = source.names().filter(|name| eligible(name)).collect();
        if expected.is_empty()
            || (q8_head && !expected.contains(&"lm_head.weight"))
            || expected.len() != manifest.matrix_count
            || checkpoint.names().count() != expected.len() * 2
        {
            return Err(OcrError::Format("quantized matrix set mismatch".into()));
        }
        let mut matrices = std::collections::BTreeMap::new();
        for name in expected {
            let original = source.tensor(name)?;
            let weight = checkpoint.tensor(name)?;
            let matrix_packed = packed_int4 && !(q8_head && name == "lm_head.weight");
            let suffix = if matrix_packed {
                ".q4_scales"
            } else {
                ".q8_scales"
            };
            let scales = checkpoint.tensor(&format!("{name}{suffix}"))?;
            if original.shape.len() != 2
                || original.shape.contains(&0)
                || !matches!(original.dtype, DType::BF16 | DType::F16 | DType::F32)
                || scales.dtype != DType::F32
                || scales.shape != [original.shape[0], original.shape[1].div_ceil(BLOCK)]
            {
                return Err(OcrError::Format(format!("invalid quantized matrix {name}")));
            }
            let (rows, cols) = (original.shape[0], original.shape[1]);
            let matrix = if matrix_packed {
                if weight.dtype != DType::U8 || weight.shape != [rows, cols.div_ceil(BLOCK), 16] {
                    return Err(OcrError::Format(format!("invalid INT4 matrix {name}")));
                }
                QuantizedMatrix::Int4(super::int4::Int4Matrix::from_parts(
                    rows,
                    cols,
                    weight.bytes.to_vec(),
                    scales.to_f32(),
                )?)
            } else {
                if weight.dtype != DType::I8
                    || weight.shape != original.shape
                    || weight.bytes.contains(&128)
                {
                    return Err(OcrError::Format(format!("invalid INT8 matrix {name}")));
                }
                QuantizedMatrix::Int8(Int8Matrix::from_parts(
                    rows,
                    cols,
                    weight.bytes.iter().map(|&v| v as i8).collect(),
                    scales.to_f32(),
                )?)
            };
            matrices.insert(name.to_owned(), matrix);
        }
        Ok(Self {
            matrices,
            packed_int4,
            q8_head,
        })
    }
    pub fn precision(&self, mode: QuantizationMode) -> &'static str {
        if self.q8_head {
            return match mode {
                QuantizationMode::WeightsOnly => "int4_body_int8_head_fp32",
                QuantizationMode::DynamicActivations => {
                    "int4_body_int8_head_dynamic_int8_activations"
                }
            };
        }
        match (self.packed_int4, mode) {
            (false, QuantizationMode::WeightsOnly) => "int8_weights_fp32",
            (false, QuantizationMode::DynamicActivations) => "int8_weights_dynamic_activations",
            (true, QuantizationMode::WeightsOnly) => "int4_weights_fp32",
            (true, QuantizationMode::DynamicActivations) => "int4_weights_dynamic_int8_activations",
        }
    }
    pub fn get(&self, name: &str) -> Option<&QuantizedMatrix> {
        self.matrices.get(name)
    }
}
