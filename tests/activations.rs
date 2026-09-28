use uocr::cpu::{Backend, CpuContext, ops};

#[test]
fn parallel_activations_preserve_scalar_bits_over_blocks_tails_and_extremes() {
    let cpu = CpuContext::new(Backend::Auto, 8, 1 << 20).unwrap();
    for len in [0, 1, 31, 1023, 16383, 16384, 16385, 200003] {
        let mut input: Vec<f32> = (0..len)
            .map(|i| ((i as f32 * 0.131).sin()) * 100.)
            .collect();
        if len > 31 {
            input[..8].copy_from_slice(&[
                0.,
                -0.,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                1e20,
            ]);
        }
        for quick in [false, true] {
            let mut expected = input.clone();
            let mut actual = input.clone();
            if quick {
                ops::quick_gelu(&mut expected);
                ops::quick_gelu_parallel(&mut actual, &cpu);
            } else {
                ops::gelu(&mut expected);
                ops::gelu_parallel(&mut actual, &cpu);
            }
            for (&got, &want) in actual.iter().zip(&expected) {
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
            }
        }
    }
}
