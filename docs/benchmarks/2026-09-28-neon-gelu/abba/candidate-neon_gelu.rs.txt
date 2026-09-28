//! Small-argument GELU using the same FP32 polynomial as libm 0.2.16.
// Polynomial and evaluation order adapted from FreeBSD s_erff.c via libm:
// Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

pub(super) fn gelu_in_place(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let (chunks, tail) = values.as_chunks_mut::<4>();
    // SAFETY: NEON is mandatory on AArch64. Each load/store accesses a complete
    // four-element array and supports unaligned storage. Excluded lanes are
    // masked before polynomial evaluation; tails use the scalar implementation.
    unsafe {
        let one = vdupq_n_f32(1.);
        let half = vdupq_n_f32(0.5);
        for chunk in chunks {
            let original = *chunk;
            let input = vld1q_f32(chunk.as_ptr());
            let scaled = vmulq_n_f32(input, std::f32::consts::FRAC_1_SQRT_2);
            let bits = vandq_u32(vreinterpretq_u32_f32(scaled), vdupq_n_u32(0x7fffffff));
            let mask = vandq_u32(
                vcgeq_u32(bits, vdupq_n_u32(0x31800000)),
                vcltq_u32(bits, vdupq_n_u32(0x3f580000)),
            );
            if vmaxvq_u32(mask) == 0 {
                super::ops::gelu(chunk);
                continue;
            }
            let x = vbslq_f32(mask, scaled, vdupq_n_f32(0.));
            let z = vmulq_f32(x, x);
            let mut r = vdupq_n_f32(f32::from_bits(0xb7c756b1));
            for bits in [0xbbbd1489, 0xbce9528f, 0xbea66beb, 0x3e0375d4] {
                r = vaddq_f32(vdupq_n_f32(f32::from_bits(bits)), vmulq_f32(z, r));
            }
            let mut s = vdupq_n_f32(f32::from_bits(0xb684e21a));
            for bits in [0x390aee49, 0x3ba68116, 0x3d852a63, 0x3ecbbbce, 0x3f800000] {
                s = vaddq_f32(vdupq_n_f32(f32::from_bits(bits)), vmulq_f32(z, s));
            }
            // Separate multiply/add instructions intentionally preserve libm's
            // rounding sequence. Full-precision division, not reciprocal estimates.
            let erf = vaddq_f32(x, vmulq_f32(x, vdivq_f32(r, s)));
            let result = vmulq_f32(input, vmulq_f32(half, vaddq_f32(one, erf)));
            vst1q_f32(chunk.as_mut_ptr(), result);
            if vminvq_u32(mask) != u32::MAX {
                let mut valid = [0_u32; 4];
                vst1q_u32(valid.as_mut_ptr(), mask);
                for i in 0..4 {
                    if valid[i] == 0 {
                        let x = original[i];
                        chunk[i] =
                            x * (0.5 * (1. + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2)));
                    }
                }
            }
        }
    }
    super::ops::gelu(tail);
}

#[cfg(test)]
mod tests {
    fn check(input: &[f32], offset: usize) {
        let mut expected = input.to_vec();
        crate::cpu::ops::gelu(&mut expected);
        let mut storage = vec![123.25; offset];
        storage.extend_from_slice(input);
        storage.extend_from_slice(&[456.75; 3]);
        super::gelu_in_place(&mut storage[offset..offset + input.len()]);
        assert!(storage[..offset].iter().all(|&x| x == 123.25));
        assert!(storage[offset + input.len()..].iter().all(|&x| x == 456.75));
        for (i, (&got, &want)) in storage[offset..offset + input.len()]
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "index={i} input={:08x} actual={:08x} expected={:08x}",
                input[i].to_bits(),
                got.to_bits(),
                want.to_bits()
            );
        }
    }

    #[test]
    fn gelu_matches_libm_bits_for_boundaries_mixed_lanes_alignment_and_tails() {
        let mut values = vec![
            0.,
            -0.,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        for boundary in [0x31800000_u32, 0x3f580000, 0x3fa00000, 0x40c00000] {
            for delta in -8_i32..=8 {
                let erf_input = f32::from_bits(boundary.wrapping_add_signed(delta));
                let input = erf_input / std::f32::consts::FRAC_1_SQRT_2;
                for neighbor in -2_i32..=2 {
                    let x = f32::from_bits(input.to_bits().wrapping_add_signed(neighbor));
                    values.extend_from_slice(&[x, -x, 0.125, -2.]);
                }
            }
        }
        values.extend((0..200001).map(|i| i as f32 / 50000. - 2.));
        let mut state = 0x713495ab_u32;
        for _ in 0..100000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            values.push(f32::from_bits(state));
        }
        for offset in 0..4 {
            check(&values, offset);
            for len in 0..17 {
                check(&values[..len], offset);
                check(&[0.125, -0.25, 0.5, -0.75].repeat(5)[..len], offset);
            }
        }
    }
}
