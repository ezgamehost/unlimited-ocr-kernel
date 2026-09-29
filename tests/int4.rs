use uocr::{
    cpu::{Backend, CpuContext, int4::Int4Matrix, quantized::Int8Matrix},
    tensor::Tensor,
};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn prepared_activation_matches_dynamic_paths_and_rejects_wrong_width() {
    use uocr::cpu::quantized::QuantizedActivation;
    for cols in [1usize, 31, 32, 33, 1280] {
        let rows = 9;
        let blocks = cols.div_ceil(32);
        let mut packed = vec![0; rows * blocks * 16];
        for row in 0..rows {
            for col in 0..cols {
                packed[(row * blocks + col / 32) * 16 + col % 16] |=
                    ((col % 7) as u8) << (4 * ((col % 32) / 16));
            }
        }
        let q4 = Int4Matrix::from_parts(rows, cols, packed, vec![0.13; rows * blocks]).unwrap();
        let q8 = Int8Matrix::quantize(&q4.dequantize().unwrap()).unwrap();
        for shape in [vec![cols], vec![1, cols], vec![1, 1, cols]] {
            let x =
                Tensor::new(shape, (0..cols).map(|i| (i as f32 * 0.17).sin()).collect()).unwrap();
            let ready = QuantizedActivation::new(&x).unwrap();
            for backend in [Backend::Scalar, Backend::Auto] {
                let mut cpu = CpuContext::new(backend, 3, 1 << 20).unwrap();
                let a = q8.linear_dynamic_int8(&x, &mut cpu).unwrap();
                let b = q8.linear_prepared(&ready, &mut cpu).unwrap();
                assert_eq!(a.shape, b.shape);
                assert_eq!(bits(&a.data), bits(&b.data));
                let a = q4.linear_dynamic_int8(&x, &mut cpu).unwrap();
                let b = q4.linear_prepared(&ready, &mut cpu).unwrap();
                assert_eq!(a.shape, b.shape);
                assert_eq!(bits(&a.data), bits(&b.data));
            }
        }
        let wrong =
            QuantizedActivation::new(&Tensor::new(vec![1, cols + 1], vec![1.; cols + 1]).unwrap())
                .unwrap();
        let mut cpu = CpuContext::new(Backend::Auto, 1, 1 << 20).unwrap();
        assert!(q8.linear_prepared(&wrong, &mut cpu).is_err());
        assert!(q4.linear_prepared(&wrong, &mut cpu).is_err());
    }
    for x in [
        Tensor::new(vec![2, 3], vec![1.; 6]).unwrap(),
        Tensor::new(vec![1, 1], vec![f32::NAN]).unwrap(),
    ] {
        assert!(QuantizedActivation::new(&x).is_err());
    }
}

#[test]
fn dynamic_projections_fit_without_activation_clone_scratch() {
    let (rows, cols) = (9, 33);
    let weights = Tensor::new(vec![rows, cols], vec![0.5; rows * cols]).unwrap();
    let q8 = Int8Matrix::quantize(&weights).unwrap();
    let mut packed = vec![0; rows * 32];
    for row in 0..rows {
        packed[row * 32..row * 32 + 16].fill(0x11);
        packed[row * 32 + 16] = 1;
    }
    let q4 = Int4Matrix::from_parts(rows, cols, packed, vec![0.5; rows * 2]).unwrap();
    for shape in [vec![cols], vec![1, cols], vec![1, 1, cols]] {
        let x = Tensor::new(shape, vec![0.25; cols]).unwrap();
        let budget = (cols + rows + cols.div_ceil(32)) * 4;
        let mut small = CpuContext::new(Backend::Auto, 1, budget).unwrap();
        let mut large = CpuContext::new(Backend::Auto, 1, 1 << 20).unwrap();
        assert_eq!(
            bits(&q8.linear_dynamic_int8(&x, &mut small).unwrap().data),
            bits(&q8.linear_dynamic_int8(&x, &mut large).unwrap().data)
        );
        assert_eq!(
            bits(&q4.linear_dynamic_int8(&x, &mut small).unwrap().data),
            bits(&q4.linear_dynamic_int8(&x, &mut large).unwrap().data)
        );
    }
}

#[test]
fn literal_nibbles_and_invalid_storage() {
    let mut bytes = vec![0; 32];
    bytes[0] = 0x79;
    bytes[1] = 0xe3;
    bytes[16] = 0x07;
    let matrix = Int4Matrix::from_parts(1, 33, bytes.clone(), vec![0.5, 2.]).unwrap();
    let restored = matrix.dequantize().unwrap();
    assert_eq!(
        (
            restored.data[0],
            restored.data[1],
            restored.data[16],
            restored.data[17],
            restored.data[32]
        ),
        (-3.5, 1.5, 3.5, -1., 14.)
    );
    assert_eq!(matrix.storage_bytes(), 40);
    for index in [0, 16] {
        let mut bad = bytes.clone();
        bad[index] = 0x08;
        assert!(Int4Matrix::from_parts(1, 33, bad, vec![1.; 2]).is_err());
    }
    bytes[16] = 0x17;
    assert!(Int4Matrix::from_parts(1, 33, bytes, vec![1.; 2]).is_err());
    for scale in [f32::NAN, f32::INFINITY, -1.] {
        assert!(Int4Matrix::from_parts(1, 1, vec![0; 16], vec![scale]).is_err());
    }
    for (rows, cols, len, scales) in [
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (1, 33, 16, 2),
        (2, 32, 32, 1),
        (usize::MAX, 32, 0, 0),
        (1, usize::MAX, 0, 0),
    ] {
        assert!(Int4Matrix::from_parts(rows, cols, vec![0; len], vec![1.; scales]).is_err());
    }
}

#[test]
fn packed_matches_simulation_for_dequantization_and_linear() {
    for cols in [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 131, 512, 513, 1280] {
        let rows = if cols == 512 || cols == 513 { 257 } else { 7 };
        let blocks = cols.div_ceil(32);
        let codes: Vec<i8> = (0..rows * cols).map(|i| (i % 15) as i8 - 7).collect();
        let scales: Vec<f32> = (0..rows * blocks)
            .map(|i| [0., -0., 0.1234567, 1e-10, 1.3][i % 5])
            .collect();
        let mut packed = vec![0u8; rows * blocks * 16];
        for row in 0..rows {
            for col in 0..cols {
                packed[(row * blocks + col / 32) * 16 + col % 16] |=
                    ((codes[row * cols + col] as u8) & 15) << (4 * ((col % 32) / 16));
            }
        }
        let q4 = Int4Matrix::from_parts(rows, cols, packed, scales.clone()).unwrap();
        let q8 = Int8Matrix::from_parts(rows, cols, codes, scales).unwrap();
        assert_eq!(
            bits(&q4.dequantize().unwrap().data),
            bits(&q8.dequantize().unwrap().data)
        );
        for backend in [Backend::Scalar, Backend::Auto] {
            let mut cpu = CpuContext::new(backend, 3, 8 << 20).unwrap();
            for batch in [1, 3] {
                let x = Tensor::new(
                    vec![batch, cols],
                    (0..batch * cols).map(|i| (i as f32 * 0.13).cos()).collect(),
                )
                .unwrap();
                let actual = q4.linear(&x, &mut cpu).unwrap();
                let expected = q8.linear(&x, &mut cpu).unwrap();
                assert_eq!(actual.shape, expected.shape);
                for (a, b) in actual.data.iter().zip(&expected.data) {
                    assert!((a - b).abs() < 1e-4 * b.abs().max(1.), "{cols}: {a} != {b}");
                }
                if batch == 1 {
                    assert_eq!(
                        bits(&q4.linear_dynamic_int8(&x, &mut cpu).unwrap().data),
                        bits(&q8.linear_dynamic_int8(&x, &mut cpu).unwrap().data)
                    );
                    let mut small = CpuContext::new(backend, 1, 1).unwrap();
                    assert!(q4.linear(&x, &mut small).is_err());
                    assert!(q4.linear_dynamic_int8(&x, &mut small).is_err());
                } else {
                    assert!(q4.linear_dynamic_int8(&x, &mut cpu).is_err());
                }
            }
        }
    }
}
