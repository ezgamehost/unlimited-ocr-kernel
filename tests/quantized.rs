use uocr::{
    cpu::{Backend, CpuContext, quantized::Int8Matrix},
    tensor::Tensor,
};
#[test]
fn int8_blocks_roundtrip_zero_rows_tails_and_linear() {
    for cols in [1, 31, 32, 33, 131, 513] {
        let rows = if cols == 513 { 257 } else { 7 };
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| {
                if i / cols == 0 {
                    0.
                } else {
                    (i as f32 * 0.37).sin() * 3.
                }
            })
            .collect();
        let matrix =
            Int8Matrix::quantize(&Tensor::new(vec![rows, cols], values.clone()).unwrap()).unwrap();
        let restored = matrix.dequantize().unwrap();
        for (a, b) in restored.data.iter().zip(&values) {
            assert!((a - b).abs() <= 3. / 254. + 1e-6);
        }
        assert!(restored.data[..cols].iter().all(|&v| v == 0.));
        for batch in [1, 3] {
            let x = Tensor::new(
                vec![batch, cols],
                (0..batch * cols).map(|i| (i as f32 * 0.13).cos()).collect(),
            )
            .unwrap();
            for backend in [Backend::Scalar, Backend::Auto] {
                let mut cpu = CpuContext::new(backend, 3, 1 << 20).unwrap();
                let output = matrix.linear(&x, &mut cpu).unwrap();
                for row in 0..batch {
                    for out in 0..rows {
                        let want: f64 = (0..cols)
                            .map(|j| {
                                x.data[row * cols + j] as f64 * restored.data[out * cols + j] as f64
                            })
                            .sum();
                        assert!((output.data[row * rows + out] as f64 - want).abs() < 2e-5);
                    }
                }
            }
        }
    }
}
#[test]
fn invalid_int8_storage_scales_and_weights_are_rejected() {
    assert!(Int8Matrix::from_parts(1, 33, vec![0; 33], vec![1.]).is_err());
    assert!(Int8Matrix::from_parts(1, 1, vec![0], vec![f32::NAN]).is_err());
    assert!(Int8Matrix::from_parts(1, 1, vec![0], vec![-1.]).is_err());
    assert!(Int8Matrix::from_parts(0, 1, vec![], vec![]).is_err());
    assert!(Int8Matrix::quantize(&Tensor::new(vec![1, 1], vec![f32::INFINITY]).unwrap()).is_err());
    let m = Int8Matrix::from_parts(1, 1, vec![1], vec![1.]).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 1, 1024).unwrap();
    assert!(
        m.linear(&Tensor::new(vec![1, 2], vec![1.; 2]).unwrap(), &mut cpu)
            .is_err()
    );
}

#[test]
fn dynamic_int8_activation_dot_matches_explicit_quantized_reference() {
    for cols in [1, 15, 31, 32, 33, 131, 1280] {
        let rows = if cols == 1280 { 129 } else { 9 };
        let matrix = Int8Matrix::quantize(
            &Tensor::new(
                vec![rows, cols],
                (0..rows * cols).map(|i| (i as f32 * 0.41).sin()).collect(),
            )
            .unwrap(),
        )
        .unwrap();
        let x = Tensor::new(
            vec![1, cols],
            (0..cols).map(|i| (i as f32 * 0.17).cos() * 2.).collect(),
        )
        .unwrap();
        let rounded = Int8Matrix::quantize(&x).unwrap().dequantize().unwrap();
        let w = matrix.dequantize().unwrap();
        let mut cpu = CpuContext::new(Backend::Auto, 3, 1 << 20).unwrap();
        let result = matrix.linear_dynamic_int8(&x, &mut cpu).unwrap();
        for row in 0..rows {
            let expected: f64 = (0..cols)
                .map(|j| w.data[row * cols + j] as f64 * rounded.data[j] as f64)
                .sum();
            assert!(
                (result.data[row] as f64 - expected).abs() < 2e-4,
                "cols={cols}"
            );
        }
    }
}
