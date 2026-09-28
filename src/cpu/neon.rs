//! ARM64 BF16 storage with FP32 fused accumulation. No BF16 ISA extension needed.
#[cfg(target_arch = "aarch64")]
pub(super) fn dot_bf16(weights: &[u8], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(weights.len(), x.len() * 2);
    // SAFETY: NEON is mandatory on AArch64. Loads are unaligned-safe; each
    // iteration checks that 16 input values and 32 weight bytes remain.
    // Byte storage is read through raw pointers without creating aligned refs.
    unsafe {
        let mut a = vdupq_n_f32(0.);
        let mut b = a;
        let mut c = a;
        let mut d = a;
        let mut i = 0;
        while i + 16 <= x.len() {
            let w0 = vld1q_u16(weights.as_ptr().add(2 * i).cast());
            let w1 = vld1q_u16(weights.as_ptr().add(2 * i + 16).cast());
            a = vfmaq_f32(
                a,
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(w0))),
                vld1q_f32(x.as_ptr().add(i)),
            );
            b = vfmaq_f32(
                b,
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(w0))),
                vld1q_f32(x.as_ptr().add(i + 4)),
            );
            c = vfmaq_f32(
                c,
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(w1))),
                vld1q_f32(x.as_ptr().add(i + 8)),
            );
            d = vfmaq_f32(
                d,
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(w1))),
                vld1q_f32(x.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
        for j in i..x.len() {
            sum += f32::from_bits(
                u32::from(u16::from_le_bytes([weights[2 * j], weights[2 * j + 1]])) << 16,
            ) * x[j];
        }
        sum
    }
}

/// Validate logits while reducing the maximum; negative infinity is a mask.
#[cfg(target_arch = "aarch64")]
pub(super) fn softmax_max(values: &[f32]) -> Option<f32> {
    use std::arch::aarch64::*;
    // SAFETY: AArch64 supplies NEON and each load stays inside a full chunk.
    unsafe {
        let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
        let mut valid = vdupq_n_u32(u32::MAX);
        let infinity = vdupq_n_f32(f32::INFINITY);
        let (chunks, tail) = values.as_chunks::<4>();
        for chunk in chunks {
            let v = vld1q_f32(chunk.as_ptr());
            valid = vandq_u32(
                valid,
                vandq_u32(vceqq_f32(v, v), vmvnq_u32(vceqq_f32(v, infinity))),
            );
            maximum = vmaxq_f32(maximum, v);
        }
        if vminvq_u32(valid) == 0 {
            return None;
        }
        let mut maximum = vmaxvq_f32(maximum);
        for &v in tail {
            if v.is_nan() || v == f32::INFINITY {
                return None;
            }
            maximum = maximum.max(v);
        }
        maximum.is_finite().then_some(maximum)
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) fn sum(values: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    // SAFETY: each load reads a complete four-float chunk of the input.
    unsafe {
        let mut sum = vdupq_n_f32(0.);
        let (chunks, tail) = values.as_chunks::<4>();
        for chunk in chunks {
            sum = vaddq_f32(sum, vld1q_f32(chunk.as_ptr()));
        }
        vaddvq_f32(sum) + tail.iter().sum::<f32>()
    }
}

/// Weight-only INT8: activations are never quantized.
#[cfg(target_arch = "aarch64")]
pub(super) fn dot_int8(weights: &[i8], x: &[f32], scales: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(weights.len(), x.len());
    assert_eq!(scales.len(), x.len().div_ceil(32));
    // SAFETY: NEON is mandatory on AArch64. Each 16-byte load is bounded by
    // its current 32-element block; tails use scalar indexing. Float loads
    // have the same bounded length, and neither input is mutated.
    unsafe {
        let mut total = vdupq_n_f32(0.);
        let mut tail = 0.;
        for (block, &scale) in scales.iter().enumerate() {
            let start = block * 32;
            let end = (start + 32).min(x.len());
            let mut a = vdupq_n_f32(0.);
            let mut b = a;
            let mut c = a;
            let mut d = a;
            let mut i = start;
            while i + 16 <= end {
                let packed = vld1q_s8(weights.as_ptr().add(i));
                let lo = vmovl_s8(vget_low_s8(packed));
                let hi = vmovl_s8(vget_high_s8(packed));
                a = vfmaq_f32(
                    a,
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                    vld1q_f32(x.as_ptr().add(i)),
                );
                b = vfmaq_f32(
                    b,
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                    vld1q_f32(x.as_ptr().add(i + 4)),
                );
                c = vfmaq_f32(
                    c,
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                    vld1q_f32(x.as_ptr().add(i + 8)),
                );
                d = vfmaq_f32(
                    d,
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                    vld1q_f32(x.as_ptr().add(i + 12)),
                );
                i += 16;
            }
            let sum = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d));
            total = vfmaq_n_f32(total, sum, scale);
            for j in i..end {
                tail += weights[j] as f32 * scale * x[j];
            }
        }
        vaddvq_f32(total) + tail
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn dot_int8_pair(weights: &[i8], x: &[i8], ws: &[f32], xs: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    assert_eq!(weights.len(), x.len());
    assert_eq!(ws.len(), x.len().div_ceil(32));
    assert_eq!(ws.len(), xs.len());
    let mut total = 0.;
    for b in 0..ws.len() {
        let start = b * 32;
        let end = (start + 32).min(x.len());
        let mut sum = vdupq_n_s32(0);
        let mut i = start;
        while i + 16 <= end {
            // SAFETY: the caller has checked dotprod support. Both loads stay
            // within their equal-length slices. A block sum is at most
            // 32*128*128, so signed 32-bit accumulation cannot overflow.
            unsafe {
                sum = vdotq_s32(
                    sum,
                    vld1q_s8(weights.as_ptr().add(i)),
                    vld1q_s8(x.as_ptr().add(i)),
                );
            }
            i += 16;
        }
        let mut sum = vaddvq_s32(sum);
        for j in i..end {
            sum += weights[j] as i32 * x[j] as i32;
        }
        total += sum as f32 * ws[b] * xs[b];
    }
    total
}

/// Four row-major outputs sharing each activation block. No weight repacking.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn dot_int8_four_rows(
    weights: &[i8],
    x: &[i8],
    ws: &[f32],
    xs: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    assert!(x.len().is_multiple_of(32));
    assert_eq!(weights.len().checked_div(4), Some(x.len()));
    assert!(weights.len().is_multiple_of(4));
    let blocks = x.len() / 32;
    assert_eq!(xs.len(), blocks);
    assert_eq!(ws.len(), blocks * 4);
    let mut total = vdupq_n_f32(0.);
    for b in 0..blocks {
        // SAFETY: caller proves dotprod availability. Every load reads exactly
        // half of a complete 32-byte block in one of four validated rows.
        // Integer block sums fit i32 even for -128 inputs. The scale array and
        // output array each contain exactly four initialized floats.
        unsafe {
            let x0 = vld1q_s8(x.as_ptr().add(b * 32));
            let x1 = vld1q_s8(x.as_ptr().add(b * 32 + 16));
            let mut sums = [vdupq_n_s32(0); 4];
            for (row, sum) in sums.iter_mut().enumerate() {
                let w = weights.as_ptr().add(row * x.len() + b * 32);
                *sum = vdotq_s32(vdotq_s32(*sum, vld1q_s8(w), x0), vld1q_s8(w.add(16)), x1);
            }
            let sums = vpaddq_s32(vpaddq_s32(sums[0], sums[1]), vpaddq_s32(sums[2], sums[3]));
            let scales = [
                ws[b],
                ws[blocks + b],
                ws[2 * blocks + b],
                ws[3 * blocks + b],
            ];
            // Deliberately separate multiplies/add: match sum * ws * xs and
            // the original per-block FP32 accumulation without FMA contraction.
            let scaled = vmulq_n_f32(
                vmulq_f32(vcvtq_f32_s32(sums), vld1q_f32(scales.as_ptr())),
                xs[b],
            );
            total = vaddq_f32(total, scaled);
        }
    }
    let mut result = [0.; 4];
    // SAFETY: result has exactly four writable floats.
    unsafe {
        vst1q_f32(result.as_mut_ptr(), total);
    }
    result
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    #[test]
    fn four_rows_match_single_row_dot_bits() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        for cols in [32usize, 64, 896, 1280] {
            let blocks = cols / 32;
            let weights: Vec<i8> = (0..cols * 4)
                .map(|i| ((i * 53 % 256) as i16 - 128) as i8)
                .collect();
            let x: Vec<i8> = (0..cols)
                .map(|i| ((i * 71 % 256) as i16 - 128) as i8)
                .collect();
            let scales: Vec<f32> = (0..blocks * 4)
                .map(|i| [0., 0.00317, 0.7, 1e-10, 17.][i % 5])
                .collect();
            let xs: Vec<f32> = (0..blocks).map(|i| (i % 17 + 1) as f32 / 123.).collect();
            // SAFETY: runtime detection above and exact complete-block storage.
            let actual = unsafe { dot_int8_four_rows(&weights, &x, &scales, &xs) };
            for row in 0..4 {
                let expected = unsafe {
                    dot_int8_pair(
                        &weights[row * cols..(row + 1) * cols],
                        &x,
                        &scales[row * blocks..(row + 1) * blocks],
                        &xs,
                    )
                };
                assert_eq!(actual[row].to_bits(), expected.to_bits());
            }
        }
    }
}
