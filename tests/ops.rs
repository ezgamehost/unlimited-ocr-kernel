use uocr::{
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};

fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor::new(shape.to_vec(), values.to_vec()).unwrap()
}

#[test]
fn linear_uses_output_by_input_weights_and_bias() {
    let x = tensor(&[2, 3], &[1., 2., 3., -1., 0., 2.]);
    let w = tensor(&[2, 3], &[1., 0., -1., 2., 1., 0.]);
    for backend in [Backend::Scalar, Backend::Auto] {
        let mut cpu = CpuContext::new(backend, 2, 1024 * 1024).unwrap();
        let y = ops::linear(&x, &w, Some(&[1., -1.]), &mut cpu).unwrap();
        assert_eq!(y.shape, [2, 2]);
        assert_eq!(y.data, [-1., 3., -2., -3.]);
    }
}

#[test]
fn rejects_invalid_shapes_and_resource_limits() {
    assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
    assert!(Tensor::new(vec![2, 2], vec![0.]).is_err());
    let mut cpu = CpuContext::new(Backend::Scalar, 1, 4).unwrap();
    let x = tensor(&[1, 2], &[1., 2.]);
    let w = tensor(&[2, 2], &[1., 0., 0., 1.]);
    assert!(ops::linear(&x, &w, None, &mut cpu).is_err());
    assert!(CpuContext::new(Backend::Scalar, 0, 100).is_err());
}

#[test]
fn softmax_is_stable_and_rejects_nonfinite_rows() {
    let mut x = [1000., 1000., -1000.];
    ops::softmax(&mut x).unwrap();
    assert_eq!(x, [0.5, 0.5, 0.]);
    assert!(ops::softmax(&mut [f32::NAN, 0.]).is_err());
    assert!(ops::softmax(&mut [f32::NEG_INFINITY; 2]).is_err());
}

#[test]
fn layernorm_and_rmsnorm_have_distinct_mean_behavior() {
    let x = tensor(&[1, 2], &[1., 3.]);
    let y = ops::layer_norm(&x, &[1., 1.], Some(&[0., 0.]), 1e-6).unwrap();
    assert!((y.data[0] + 1.).abs() < 1e-5);
    assert!((y.data[1] - 1.).abs() < 1e-5);
    let y = ops::rms_norm(&x, &[1., 1.], 1e-6).unwrap();
    assert!((y.data[0] - 0.44721355).abs() < 1e-6);
    assert!((y.data[1] - 1.3416406).abs() < 1e-6);
}

#[test]
fn conv_nhwc_matches_hand_computed_patch_sums() {
    let x = tensor(&[1, 3, 3, 1], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let w = tensor(&[1, 1, 2, 2], &[1., 1., 1., 1.]);
    let mut cpu = CpuContext::new(Backend::Auto, 1, 1 << 20).unwrap();
    let y = ops::conv2d(&x, &w, None, 1, 0, &mut cpu).unwrap();
    assert_eq!(y.shape, [1, 2, 2, 1]);
    assert_eq!(y.data, [12., 16., 24., 28.]);
}

#[test]
fn rotary_uses_split_halves_and_absolute_positions() {
    let mut values: [f32; 4] = [1., 2., 3., 4.];
    ops::rope(&mut values, 4, 0, 10000.).unwrap();
    assert_eq!(values, [1., 2., 3., 4.]);
    ops::rope(&mut values, 4, 1, 10000.).unwrap();
    let want = [-1.9841106, 1.9599007, 2.462378, 4.0197997];
    for (a, b) in values.iter().zip(want) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn activations_match_known_values() {
    let mut x = [-1.0_f32, 0., 1.];
    ops::silu(&mut x);
    for (a, b) in x.iter().zip([-0.26894143, 0., 0.7310586]) {
        assert!((a - b).abs() < 1e-6);
    }
    let mut x = [-1.0_f32, 0., 1.];
    ops::gelu(&mut x);
    for (a, b) in x.iter().zip([-0.15865526, 0., 0.8413447]) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn causal_attention_excludes_future_values() {
    let q = tensor(&[1, 2, 2], &[0., 0., 0., 0.]);
    let k = q.clone();
    let v = tensor(&[1, 2, 2], &[2., 4., 6., 8.]);
    let mut cpu = CpuContext::new(Backend::Auto, 1, 1 << 20).unwrap();
    let y = ops::attention(&q, &k, &v, true, &mut cpu).unwrap();
    assert_eq!(y.data, [2., 4., 4., 6.]);
    let y = ops::attention(&q, &k, &v, false, &mut cpu).unwrap();
    assert_eq!(y.data, [4., 6., 4., 6.]);
}

#[test]
fn accelerated_softmax_matches_f64_oracle_for_masked_and_long_rows() {
    for len in [1, 3, 17, 196, 4096] {
        let input: Vec<f32> = (0..len)
            .map(|i| {
                if i % 7 == 3 {
                    f32::NEG_INFINITY
                } else {
                    1000. + (i as f32 * 0.37).sin() * 30.
                }
            })
            .collect();
        let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp: Vec<f64> = input.iter().map(|&x| (x as f64 - maximum).exp()).collect();
        let sum: f64 = exp.iter().sum();
        for backend in [Backend::Scalar, Backend::Auto] {
            let mut row = input.clone();
            ops::softmax_backend(&mut row, backend).unwrap();
            for (actual, expected) in row.iter().zip(&exp) {
                assert!((*actual as f64 - expected / sum).abs() < 2e-6);
            }
        }
    }
    for mut row in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
    ] {
        assert!(ops::softmax_backend(&mut row, Backend::Auto).is_err());
    }
}

#[test]
fn vector_softmax_rejects_invalid_values_in_every_lane_and_tail() {
    for position in 0..19 {
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut row = vec![0.; 19];
            row[position] = invalid;
            assert!(ops::softmax_backend(&mut row, Backend::Auto).is_err());
        }
    }
    assert!(ops::softmax_backend(&mut [f32::NEG_INFINITY; 19], Backend::Auto).is_err());
}

#[test]
fn layernorm_centers_nearly_constant_wide_rows() {
    let width = 3072;
    let data: Vec<f32> = (0..width)
        .map(|i| if i % 2 == 0 { 1.0 } else { 1.0001 })
        .collect();
    let x = Tensor::new(vec![1, width], data).unwrap();
    let y = ops::layer_norm(&x, &vec![1.; width], None, 1e-6).unwrap();
    let mean = (1.0_f64 + 1.0001_f32 as f64) / 2.;
    let deviation = 1.0001_f32 as f64 - mean;
    let expected = deviation / (deviation * deviation + 1e-6_f32 as f64).sqrt();
    for (i, &actual) in y.data.iter().enumerate() {
        let expected = if i % 2 == 0 { -expected } else { expected };
        assert!(
            (actual as f64 - expected).abs() < 1e-6,
            "{actual} versus {expected}"
        );
    }
}
