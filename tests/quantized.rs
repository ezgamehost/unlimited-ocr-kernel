use uocr::{
    cpu::{Backend, CpuContext, quantized::Int8Matrix},
    tensor::Tensor,
};

#[test]
fn dequantization_matches_correctly_rounded_products_across_row_tails() {
    for cols in [1usize, 31, 32, 33, 513, 1280] {
        let rows = 5;
        let weights: Vec<i8> = (0..rows * cols)
            .map(|i| ((i % 255) as i16 - 127) as i8)
            .collect();
        let scales: Vec<f32> = (0..rows * cols.div_ceil(32))
            .map(|i| [0., -0., 0.1234567, 1e-20, 1e20][i % 5])
            .collect();
        let matrix = Int8Matrix::from_parts(rows, cols, weights.clone(), scales.clone()).unwrap();
        let restored = matrix.dequantize().unwrap();
        for row in 0..rows {
            for col in 0..cols {
                let expected = (weights[row * cols + col] as f64
                    * scales[row * cols.div_ceil(32) + col / 32] as f64)
                    as f32;
                assert_eq!(
                    restored.data[row * cols + col].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }
}
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

#[test]
fn quantization_preserves_scalar_products_across_full_blocks_and_tails() {
    for cols in [1usize, 31, 32, 33, 63, 64, 65, 896, 1280] {
        let rows = 3;
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 71 % 257) as f32 - 128.) / 37.)
            .collect();
        let input = Tensor::new(vec![rows, cols], values.clone()).unwrap();
        let actual = Int8Matrix::quantize(&input).unwrap().dequantize().unwrap();
        for (row, restored) in values
            .chunks_exact(cols)
            .zip(actual.data.chunks_exact(cols))
        {
            for (block, restored) in row.chunks(32).zip(restored.chunks(32)) {
                let scale = block.iter().map(|v| v.abs()).fold(0., f32::max) / 127.;
                for (&value, &actual) in block.iter().zip(restored) {
                    let code = if scale == 0. {
                        0
                    } else {
                        (value / scale).round_ties_even().clamp(-127., 127.) as i8
                    };
                    assert_eq!(actual.to_bits(), (code as f32 * scale).to_bits());
                }
            }
        }
    }
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ] {
        assert!(Int8Matrix::quantize(&Tensor::new(vec![1, 32], vec![bad; 32]).unwrap()).is_err());
    }
}
