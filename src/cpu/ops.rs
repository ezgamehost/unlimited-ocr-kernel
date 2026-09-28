use super::{Backend, CpuContext, accelerate};
use crate::checkpoint::{DType, WeightView};
use crate::{
    error::{OcrError, Result},
    tensor::{Tensor, element_count},
};
use rayon::prelude::*;

pub fn linear(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&[f32]>,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    x.validate()?;
    weight.validate()?;
    let k = *x
        .shape
        .last()
        .ok_or_else(|| OcrError::Shape("linear input must have a dimension".into()))?;
    if k == 0 || weight.shape.len() != 2 || weight.shape[1] != k || weight.shape[0] == 0 {
        return Err(OcrError::Shape(
            "linear weight must have shape [out,in]".into(),
        ));
    }
    let n = weight.shape[0];
    let m = x.data.len() / k;
    if bias.is_some_and(|b| b.len() != n) {
        return Err(OcrError::Shape("linear bias length mismatch".into()));
    }
    let count = m
        .checked_mul(n)
        .ok_or_else(|| OcrError::Shape("linear output overflow".into()))?;
    cpu.check_elements(count)?;
    let mut data = vec![0.; count];
    if m != 0 && cfg!(target_os = "macos") && cpu.backend != Backend::Scalar {
        accelerate::linear(&x.data, &weight.data, &mut data, m, n, k)?;
    } else {
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = x.data[i * k..(i + 1) * k]
                    .iter()
                    .zip(&weight.data[j * k..(j + 1) * k])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
    }
    if let Some(bias) = bias {
        for row in data.chunks_exact_mut(n) {
            for (v, b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
    }
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = n;
    Tensor::new(shape, data)
}

/// Direct BF16 GEMV avoids a full FP32 model copy during autoregressive decode.
pub fn linear_weight(
    x: &Tensor,
    weight: WeightView<'_>,
    bias: Option<&[f32]>,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    x.validate()?;
    weight.validate()?;
    let k = *x
        .shape
        .last()
        .ok_or_else(|| OcrError::Shape("linear input rank".into()))?;
    if weight.shape.len() != 2 || k == 0 || weight.shape[1] != k || weight.shape[0] == 0 {
        return Err(OcrError::Shape("linear weight must be [out,in]".into()));
    }
    let n = weight.shape[0];
    if bias.is_some_and(|b| b.len() != n) {
        return Err(OcrError::Shape("linear bias length".into()));
    }
    if weight.dtype == DType::I64 {
        return Err(OcrError::Unsupported("integer linear weight".into()));
    }
    if x.data.len() == k && weight.dtype == DType::BF16 {
        cpu.check_elements(n)?;
        let mut data = vec![0.; n];
        let compute = |row: usize| -> f32 {
            #[cfg(target_arch = "aarch64")]
            if cpu.backend != Backend::Scalar {
                return super::neon::dot_bf16(
                    &weight.bytes[row * k * 2..(row + 1) * k * 2],
                    &x.data,
                ) + bias.map_or(0., |b| b[row]);
            }
            weight.bytes[row * k * 2..(row + 1) * k * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&x.data)
                .map(|(b, x)| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16) * x)
                .sum::<f32>()
                + bias.map_or(0., |b| b[row])
        };
        if cpu.backend == Backend::Scalar || cpu.threads() == 1 {
            for (row, y) in data.iter_mut().enumerate() {
                *y = compute(row);
            }
        } else {
            cpu.pool.install(|| {
                data.par_iter_mut()
                    .enumerate()
                    .for_each(|(row, y)| *y = compute(row))
            });
        }
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = n;
        Tensor::new(shape, data)
    } else {
        cpu.check_elements(
            weight
                .len()
                .checked_add(x.data.len() / k * n)
                .ok_or_else(|| OcrError::Resource("linear allocation overflow".into()))?,
        )?;
        linear(
            x,
            &Tensor::new(weight.shape.to_vec(), weight.to_f32())?,
            bias,
            cpu,
        )
    }
}

pub fn softmax(values: &mut [f32]) -> Result<()> {
    softmax_backend(values, Backend::Scalar)
}

pub fn softmax_backend(values: &mut [f32], backend: Backend) -> Result<()> {
    let scalar_max = || {
        if values.is_empty() || values.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
            return None;
        }
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        maximum.is_finite().then_some(maximum)
    };
    #[cfg(target_arch = "aarch64")]
    let maximum = if backend == Backend::Scalar {
        scalar_max()
    } else {
        super::neon::softmax_max(values)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let maximum = scalar_max();
    let maximum =
        maximum.ok_or_else(|| OcrError::Numerical("invalid or fully masked softmax row".into()))?;
    for x in values.iter_mut() {
        *x -= maximum;
    }
    #[cfg(target_os = "macos")]
    if backend != Backend::Scalar {
        accelerate::exp_in_place(values);
    } else {
        for x in values.iter_mut() {
            *x = x.exp();
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        for x in values.iter_mut() {
            *x = x.exp();
        }
    }
    #[cfg(target_arch = "aarch64")]
    let sum: f32 = if backend == Backend::Scalar {
        values.iter().sum()
    } else {
        super::neon::sum(values)
    };
    #[cfg(not(target_arch = "aarch64"))]
    let sum: f32 = values.iter().sum();
    for x in values.iter_mut() {
        *x /= sum;
    }
    Ok(())
}

pub fn silu(values: &mut [f32]) {
    for value in values {
        *value /= 1. + (-*value).exp();
    }
}

pub fn gelu(values: &mut [f32]) {
    for value in values {
        *value *= 0.5 * (1. + libm::erff(*value * std::f32::consts::FRAC_1_SQRT_2));
    }
}

pub fn quick_gelu(values: &mut [f32]) {
    for value in values {
        *value /= 1. + (-1.702 * *value).exp();
    }
}

/// Attention in [heads,tokens,head_dim] layout, using bounded query blocks.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    attention_bias(q, k, v, causal, cpu, |_, _, _, _| {})
}

/// Bias callback receives head, first query, query count, and unnormalized scores.
pub fn attention_bias<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cpu: &mut CpuContext,
    mut bias: F,
) -> Result<Tensor>
where
    F: FnMut(usize, usize, usize, &mut [f32]),
{
    q.validate()?;
    k.validate()?;
    v.validate()?;
    if q.shape.len() != 3
        || k.shape.len() != 3
        || v.shape != k.shape
        || q.shape.contains(&0)
        || k.shape.contains(&0)
        || q.shape[0] != k.shape[0]
        || q.shape[2] != k.shape[2]
        || (causal && q.shape[1] != k.shape[1])
    {
        return Err(OcrError::Shape("invalid attention layout".into()));
    }
    let (heads, nq, nk, d) = (q.shape[0], q.shape[1], k.shape[1], q.shape[2]);
    let block = nq.min(32);
    let scratch = element_count(&[block, nk])?
        .checked_add(element_count(&[d, nk])?)
        .and_then(|n| n.checked_add(q.data.len()))
        .ok_or_else(|| OcrError::Resource("attention size overflow".into()))?;
    cpu.check_elements(scratch)?;
    let scale = (d as f32).sqrt().recip();
    let mut output = vec![0.; q.data.len()];
    let fast = cfg!(target_os = "macos") && cpu.backend != Backend::Scalar;
    for head in 0..heads {
        let keys = &k.data[head * nk * d..(head + 1) * nk * d];
        let values = &v.data[head * nk * d..(head + 1) * nk * d];
        let mut vt = vec![0.; d * nk];
        for i in 0..nk {
            for j in 0..d {
                vt[j * nk + i] = values[i * d + j];
            }
        }
        for start in (0..nq).step_by(block) {
            let count = block.min(nq - start);
            let queries = &q.data[(head * nq + start) * d..(head * nq + start + count) * d];
            let mut scores = vec![0.; count * nk];
            if fast {
                accelerate::linear(queries, keys, &mut scores, count, nk, d)?;
            } else {
                for i in 0..count {
                    for j in 0..nk {
                        scores[i * nk + j] = queries[i * d..(i + 1) * d]
                            .iter()
                            .zip(&keys[j * d..(j + 1) * d])
                            .map(|(a, b)| a * b)
                            .sum();
                    }
                }
            }
            for value in &mut scores {
                *value *= scale;
            }
            bias(head, start, count, &mut scores);
            for i in 0..count {
                if causal {
                    scores[i * nk + start + i + 1..(i + 1) * nk].fill(f32::NEG_INFINITY);
                }
                softmax_backend(&mut scores[i * nk..(i + 1) * nk], cpu.backend)?;
            }
            let dest = &mut output[(head * nq + start) * d..(head * nq + start + count) * d];
            if fast {
                accelerate::linear(&scores, &vt, dest, count, d, nk)?;
            } else {
                for i in 0..count {
                    for j in 0..d {
                        dest[i * d + j] = scores[i * nk..(i + 1) * nk]
                            .iter()
                            .zip(&vt[j * nk..(j + 1) * nk])
                            .map(|(a, b)| a * b)
                            .sum();
                    }
                }
            }
        }
    }
    Tensor::new(q.shape.clone(), output)
}

fn norm(
    x: &Tensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    center: bool,
) -> Result<Tensor> {
    x.validate()?;
    let width = *x
        .shape
        .last()
        .ok_or_else(|| OcrError::Shape("normalization needs a dimension".into()))?;
    if width == 0
        || weight.len() != width
        || bias.is_some_and(|b| b.len() != width)
        || !eps.is_finite()
        || eps <= 0.
    {
        return Err(OcrError::Shape("invalid normalization parameters".into()));
    }
    if x.data
        .iter()
        .chain(weight)
        .chain(bias.unwrap_or(&[]))
        .any(|v| !v.is_finite())
    {
        return Err(OcrError::Numerical("normalization".into()));
    }
    let mut data = vec![0.; x.data.len()];
    for (input, output) in x.data.chunks_exact(width).zip(data.chunks_exact_mut(width)) {
        if center {
            // Wide, nearly constant rows lose their mean under sequential FP32
            // summation. Retain precision through centering and variance.
            let mean = input.iter().map(|&v| v as f64).sum::<f64>() / width as f64;
            let variance = input
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            let scale = (variance + eps as f64).sqrt().recip();
            for i in 0..width {
                output[i] = ((input[i] as f64 - mean) * scale * weight[i] as f64
                    + bias.map_or(0., |b| b[i]) as f64) as f32;
            }
            continue;
        }
        let mean = 0.;
        let variance = input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let scale = (variance + eps).sqrt().recip();
        for i in 0..width {
            output[i] = (input[i] - mean) * scale * weight[i] + bias.map_or(0., |b| b[i]);
        }
    }
    Tensor::new(x.shape.clone(), data)
}

pub fn layer_norm(x: &Tensor, w: &[f32], b: Option<&[f32]>, eps: f32) -> Result<Tensor> {
    norm(x, w, b, eps, true)
}
pub fn rms_norm(x: &Tensor, w: &[f32], eps: f32) -> Result<Tensor> {
    norm(x, w, None, eps, false)
}

pub fn rope(values: &mut [f32], head_dim: usize, position: usize, theta: f32) -> Result<()> {
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !values.len().is_multiple_of(head_dim)
        || !theta.is_finite()
        || theta <= 0.
    {
        return Err(OcrError::Shape("invalid rotary dimensions or base".into()));
    }
    for head in values.chunks_exact_mut(head_dim) {
        for i in 0..head_dim / 2 {
            let angle = position as f32 / theta.powf(2. * i as f32 / head_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (head[i], head[i + head_dim / 2]);
            head[i] = a * cos - b * sin;
            head[i + head_dim / 2] = b * cos + a * sin;
        }
    }
    Ok(())
}

/// Input NHWC; weights OIHW, matching PyTorch convolution checkpoints.
pub fn conv2d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&[f32]>,
    stride: usize,
    pad: usize,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    x.validate()?;
    w.validate()?;
    if x.shape.len() != 4
        || w.shape.len() != 4
        || stride == 0
        || x.shape.contains(&0)
        || w.shape.contains(&0)
        || x.shape[3] != w.shape[1]
    {
        return Err(OcrError::Shape("invalid convolution dimensions".into()));
    }
    let (batch, h, width, channels) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (outputs, kh, kw) = (w.shape[0], w.shape[2], w.shape[3]);
    let padded = |size: usize| size.checked_add(pad.checked_mul(2)?);
    let oh = padded(h)
        .and_then(|v| v.checked_sub(kh))
        .map(|v| v / stride + 1)
        .ok_or_else(|| OcrError::Shape("convolution height overflow/underflow".into()))?;
    let ow = padded(width)
        .and_then(|v| v.checked_sub(kw))
        .map(|v| v / stride + 1)
        .ok_or_else(|| OcrError::Shape("convolution width overflow/underflow".into()))?;
    let rows = element_count(&[batch, oh, ow])?;
    let columns = element_count(&[channels, kh, kw])?;
    let count = element_count(&[rows, columns])?;
    cpu.check_elements(
        count
            .checked_add(element_count(&[rows, outputs])?)
            .ok_or_else(|| OcrError::Resource("convolution scratch overflow".into()))?,
    )?;
    let mut patches = vec![0.; count];
    for b in 0..batch {
        for y in 0..oh {
            for xx in 0..ow {
                let row = (b * oh + y) * ow + xx;
                for c in 0..channels {
                    for dy in 0..kh {
                        for dx in 0..kw {
                            let sy = (y * stride + dy).checked_sub(pad);
                            let sx = (xx * stride + dx).checked_sub(pad);
                            if let (Some(sy), Some(sx)) = (sy, sx)
                                && sy < h
                                && sx < width
                            {
                                patches[row * columns + (c * kh + dy) * kw + dx] =
                                    x.data[((b * h + sy) * width + sx) * channels + c];
                            }
                        }
                    }
                }
            }
        }
    }
    let patch = Tensor::new(vec![rows, columns], patches)?;
    let weight = Tensor::new(vec![outputs, columns], w.data.clone())?;
    let mut output = linear(&patch, &weight, bias, cpu)?;
    output.shape = vec![batch, oh, ow, outputs];
    Ok(output)
}
