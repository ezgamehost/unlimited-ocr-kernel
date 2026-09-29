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
    if weight.shape.len() != 2 {
        return Err(OcrError::Shape(
            "linear weight must have shape [out,in]".into(),
        ));
    }
    linear_slice(x, &weight.data, weight.shape[0], weight.shape[1], bias, cpu)
}

/// Borrow an already contiguous output-by-input matrix, including flattened
/// convolution weights, without allocating a duplicate tensor.
fn linear_slice(
    x: &Tensor,
    weight: &[f32],
    n: usize,
    columns: usize,
    bias: Option<&[f32]>,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    x.validate()?;
    let k = *x
        .shape
        .last()
        .ok_or_else(|| OcrError::Shape("linear input must have a dimension".into()))?;
    if k == 0 || columns != k || n == 0 || weight.len() != element_count(&[n, columns])? {
        return Err(OcrError::Shape(
            "linear weight must have shape [out,in]".into(),
        ));
    }
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
        accelerate::linear(&x.data, weight, &mut data, m, n, k)?;
    } else {
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = x.data[i * k..(i + 1) * k]
                    .iter()
                    .zip(&weight[j * k..(j + 1) * k])
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
    if matches!(weight.dtype, DType::I64 | DType::I8 | DType::U8) {
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
        if cpu.backend == Backend::Scalar || cpu.threads() == 1 || weight.len() <= 128 * 1024 {
            for (row, y) in data.iter_mut().enumerate() {
                *y = compute(row);
            }
        } else {
            cpu.pool.install(|| {
                data.par_iter_mut()
                    .with_min_len(n.div_ceil(cpu.threads()))
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

/// Normalize independent score rows, optionally masking beyond each absolute
/// causal query position. Parallelism never changes a row's reduction order.
pub fn softmax_rows(
    values: &mut [f32],
    width: usize,
    causal_start: Option<usize>,
    cpu: &CpuContext,
) -> Result<()> {
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(OcrError::Shape("invalid softmax row layout".into()));
    }
    let rows = values.len() / width;
    if causal_start.is_some_and(|start| start.checked_add(rows).is_none_or(|end| end > width)) {
        return Err(OcrError::Shape(
            "causal softmax position exceeds row width".into(),
        ));
    }
    let normalize = |(i, row): (usize, &mut [f32])| {
        if let Some(start) = causal_start {
            row[start + i + 1..].fill(f32::NEG_INFINITY);
        }
        softmax_backend(row, cpu.backend)
    };
    if cpu.backend != Backend::Scalar && cpu.threads() > 1 && values.len() >= 65536 {
        cpu.pool.install(|| {
            values
                .par_chunks_mut(width)
                .with_min_len(rows.div_ceil(cpu.threads()))
                .enumerate()
                .try_for_each(normalize)
        })
    } else {
        values.chunks_mut(width).enumerate().try_for_each(normalize)
    }
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

fn parallel_activation(values: &mut [f32], cpu: &CpuContext, operation: fn(&mut [f32])) {
    if cpu.backend == Backend::Scalar || cpu.threads() == 1 || values.len() < 16384 {
        operation(values);
    } else {
        let chunk = values.len().div_ceil(cpu.threads());
        cpu.pool
            .install(|| values.par_chunks_mut(chunk).for_each(operation));
    }
}

/// Preserve scalar erf arithmetic while distributing independent elements.
pub fn gelu_parallel(values: &mut [f32], cpu: &CpuContext) {
    #[cfg(target_arch = "aarch64")]
    if cpu.backend != Backend::Scalar {
        parallel_activation(values, cpu, super::neon_gelu::gelu_in_place);
        return;
    }
    parallel_activation(values, cpu, gelu);
}

pub fn quick_gelu_parallel(values: &mut [f32], cpu: &CpuContext) {
    parallel_activation(values, cpu, quick_gelu);
}

/// Attention in [heads,tokens,head_dim] layout, using bounded query blocks.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    // Keep unbiased CLIP/decoder tiling stable: larger tiles can select a
    // different BLAS reduction path and perturb generated box coordinates.
    attention_bias_blocked(q, k, v, causal, cpu, 128, |_, _, _, _| {})
}

/// Bias callback receives head, first query, query count, and unnormalized scores.
pub fn attention_bias<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cpu: &mut CpuContext,
    bias: F,
) -> Result<Tensor>
where
    F: FnMut(usize, usize, usize, &mut [f32]),
{
    // SAM benefits from larger tiles; the shared implementation still shrinks
    // them to fit the configured scratch limit.
    attention_bias_blocked(q, k, v, causal, cpu, 1024, bias)
}

fn attention_bias_blocked<F>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cpu: &mut CpuContext,
    query_cap: usize,
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
    let fixed = element_count(&[d, nk])?
        .checked_add(q.data.len())
        .ok_or_else(|| OcrError::Resource("attention size overflow".into()))?;
    cpu.check_elements(
        fixed
            .checked_add(nk)
            .ok_or_else(|| OcrError::Resource("attention scratch overflow".into()))?,
    )?;
    let block = nq.min(query_cap).min((cpu.scratch_limit / 4 - fixed) / nk);
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
            softmax_rows(&mut scores, nk, causal.then_some(start), cpu)?;
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

#[derive(Clone, Copy)]
enum Normalization {
    Layer,
    RmsSequential,
    RmsPairwise,
}

fn pairwise_square_sum(input: &[f32]) -> f32 {
    if input.len() <= 16 {
        input.iter().map(|v| v * v).sum()
    } else {
        let (left, right) = input.split_at(input.len() / 2);
        pairwise_square_sum(left) + pairwise_square_sum(right)
    }
}

fn norm(
    x: &Tensor,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    kind: Normalization,
    cpu: Option<&CpuContext>,
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
    let normalize = |(input, output): (&[f32], &mut [f32])| {
        if matches!(kind, Normalization::Layer) {
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
            return;
        }
        let mean = 0.;
        let sum = if matches!(kind, Normalization::RmsPairwise) {
            pairwise_square_sum(input)
        } else {
            input.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
        };
        let variance = sum / width as f32;
        let scale = (variance + eps).sqrt().recip();
        for i in 0..width {
            output[i] = (input[i] - mean) * scale * weight[i] + bias.map_or(0., |b| b[i]);
        }
    };
    if let Some(cpu) = cpu
        .filter(|cpu| cpu.backend != Backend::Scalar && cpu.threads() > 1 && x.data.len() >= 16384)
    {
        let rows_per_task = (x.data.len() / width).div_ceil(cpu.threads());
        cpu.pool.install(|| {
            x.data
                .par_chunks_exact(width)
                .zip(data.par_chunks_exact_mut(width))
                .with_min_len(rows_per_task)
                .for_each(normalize)
        });
    } else {
        x.data
            .chunks_exact(width)
            .zip(data.chunks_exact_mut(width))
            .for_each(normalize);
    }
    Tensor::new(x.shape.clone(), data)
}

pub fn layer_norm(x: &Tensor, w: &[f32], b: Option<&[f32]>, eps: f32) -> Result<Tensor> {
    norm(x, w, b, eps, Normalization::Layer, None)
}
/// Distribute independent rows without changing each row's reduction order.
pub fn layer_norm_parallel(
    x: &Tensor,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f32,
    cpu: &CpuContext,
) -> Result<Tensor> {
    norm(x, w, b, eps, Normalization::Layer, Some(cpu))
}
pub fn rms_norm(x: &Tensor, w: &[f32], eps: f32) -> Result<Tensor> {
    norm(x, w, None, eps, Normalization::RmsSequential, None)
}

/// RMS normalization with a balanced FP32 reduction for wide unquantized rows.
/// The sequential variant remains available for quantized output compatibility.
pub fn rms_norm_pairwise(x: &Tensor, w: &[f32], eps: f32) -> Result<Tensor> {
    norm(x, w, None, eps, Normalization::RmsPairwise, None)
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

/// Rotate queries and keys with one coefficient calculation per channel pair.
/// Heads share absolute position and base; each output retains the scalar formula.
pub(crate) fn rope_pair(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !q.len().is_multiple_of(head_dim)
        || q.len() != k.len()
        || !theta.is_finite()
        || theta <= 0.
    {
        return Err(OcrError::Shape(
            "invalid paired rotary dimensions or base".into(),
        ));
    }
    for i in 0..head_dim / 2 {
        let angle = position as f32 / theta.powf(2. * i as f32 / head_dim as f32);
        let (sin, cos) = angle.sin_cos();
        for values in [&mut *q, &mut *k] {
            for head in values.chunks_exact_mut(head_dim) {
                let (a, b) = (head[i], head[i + head_dim / 2]);
                head[i] = a * cos - b * sin;
                head[i + head_dim / 2] = b * cos + a * sin;
            }
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
    let mut output = linear_slice(&patch, &w.data, outputs, columns, bias, cpu)?;
    output.shape = vec![batch, oh, ow, outputs];
    Ok(output)
}

/// Single-query attention directly over token-major prefix/ring storage.
/// Physical ring order is immaterial: corresponding K/V entries stay paired.
pub fn attention_decode(
    q: &Tensor,
    keys: &[f32],
    values: &[f32],
    heads: usize,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    attention_decode_layout(q, keys, values, heads, None, cpu)
}

/// Head-major storage with fixed capacity per head and an explicit active count.
pub fn attention_decode_head_major(
    q: &Tensor,
    keys: &[f32],
    values: &[f32],
    heads: usize,
    tokens: usize,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    attention_decode_layout(q, keys, values, heads, Some(tokens), cpu)
}

fn attention_decode_layout(
    q: &Tensor,
    keys: &[f32],
    values: &[f32],
    heads: usize,
    active: Option<usize>,
    cpu: &mut CpuContext,
) -> Result<Tensor> {
    q.validate()?;
    let width = q.data.len();
    if heads == 0
        || width == 0
        || q.shape != [1, width]
        || !width.is_multiple_of(heads)
        || keys.is_empty()
        || !keys.len().is_multiple_of(width)
        || keys.len() != values.len()
    {
        return Err(OcrError::Shape("decode attention layout".into()));
    }
    let capacity = keys.len() / width;
    let tokens = active.unwrap_or(capacity);
    if tokens == 0 || tokens > capacity {
        return Err(OcrError::Shape("decode attention active tokens".into()));
    }
    let dim = width / heads;
    let (head_stride, token_stride) = if active.is_some() {
        (capacity * dim, dim)
    } else {
        (dim, width)
    };
    cpu.check_elements(
        tokens
            .checked_add(width)
            .ok_or_else(|| OcrError::Resource("decode attention scratch overflow".into()))?,
    )?;
    let mut output = vec![0.; width];
    let scale = (dim as f32).sqrt().recip();
    let fast = cfg!(target_os = "macos") && cpu.backend != Backend::Scalar;
    let compute_head = |head: usize, scores: &mut [f32], dest: &mut [f32]| -> Result<()> {
        let offset = head * head_stride;
        let query = &q.data[head * dim..(head + 1) * dim];
        if fast {
            accelerate::matvec_strided(
                &keys[offset..],
                query,
                scores,
                tokens,
                dim,
                token_stride,
                false,
            )?;
        } else {
            for (token, score) in scores.iter_mut().enumerate() {
                *score = query
                    .iter()
                    .zip(&keys[token * token_stride + offset..token * token_stride + offset + dim])
                    .map(|(q, k)| q * k)
                    .sum();
            }
        }
        for score in scores.iter_mut() {
            *score *= scale;
        }
        softmax_backend(scores, cpu.backend)?;
        if fast {
            accelerate::matvec_strided(
                &values[offset..],
                scores,
                dest,
                tokens,
                dim,
                token_stride,
                true,
            )?;
        } else {
            for (token, &score) in scores.iter().enumerate() {
                for (out, &value) in dest.iter_mut().zip(
                    &values[token * token_stride + offset..token * token_stride + offset + dim],
                ) {
                    *out += score * value;
                }
            }
        }
        Ok(())
    };
    let parallel_scores = tokens.checked_mul(heads);
    let parallel = cpu.threads() > 1
        && heads > 1
        && (!fast || accelerate::parallel_matvec_supported())
        && cpu.backend != Backend::Scalar
        && tokens * width >= 65536
        && parallel_scores
            .and_then(|n| n.checked_add(width))
            .is_some_and(|n| cpu.check_elements(n).is_ok());
    if parallel {
        // Preallocate exactly the checked aggregate score storage. Heads retain
        // independent arithmetic; no cross-head floating-point reduction.
        let mut scores = vec![0.; parallel_scores.expect("checked score count")];
        cpu.pool.install(|| {
            output
                .par_chunks_mut(dim)
                .zip(scores.par_chunks_mut(tokens))
                .enumerate()
                .try_for_each(|(head, (dest, scores))| compute_head(head, scores, dest))
        })?;
    } else {
        // Keep the original scratch requirement for small-budget requests.
        let mut scores = vec![0.; tokens];
        for (head, dest) in output.chunks_mut(dim).enumerate() {
            compute_head(head, &mut scores, dest)?;
        }
    }
    Tensor::new(q.shape.clone(), output)
}

#[cfg(test)]
mod rope_pair_tests {
    use super::*;

    #[test]
    fn paired_rotary_matches_independent_calls_bitwise() {
        for dim in [2, 4, 128, 256] {
            for heads in [0, 1, 2, 10] {
                for position in [0, 1, 127, 128, 1517, 8192, 131071, usize::MAX] {
                    for theta in [1., 10000., 500000.] {
                        let q: Vec<f32> = (0..dim * heads)
                            .map(|i| [0., -0., 0.00317, -0.7, 1e-10, 17., -23.][i % 7])
                            .collect();
                        let k: Vec<f32> = q.iter().rev().copied().collect();
                        let (mut expected_q, mut expected_k) = (q.clone(), k.clone());
                        rope(&mut expected_q, dim, position, theta).unwrap();
                        rope(&mut expected_k, dim, position, theta).unwrap();
                        let (mut actual_q, mut actual_k) = (q, k);
                        rope_pair(&mut actual_q, &mut actual_k, dim, position, theta).unwrap();
                        for (actual, expected) in actual_q
                            .iter()
                            .chain(&actual_k)
                            .zip(expected_q.iter().chain(&expected_k))
                        {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "dim={dim}, heads={heads}, pos={position}, theta={theta}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paired_rotary_rejects_invalid_layout_before_mutating() {
        for (q_len, k_len, dim, theta) in [
            (4, 4, 0, 10000.),
            (4, 4, 3, 10000.),
            (4, 3, 2, 10000.),
            (3, 3, 2, 10000.),
            (4, 4, 2, 0.),
            (4, 4, 2, -1.),
            (4, 4, 2, f32::NAN),
            (4, 4, 2, f32::INFINITY),
        ] {
            let mut q = vec![1.; q_len];
            let mut k = vec![2.; k_len];
            assert!(rope_pair(&mut q, &mut k, dim, 13, theta).is_err());
            assert_eq!(q, vec![1.; q_len]);
            assert_eq!(k, vec![2.; k_len]);
        }
    }
}
