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
