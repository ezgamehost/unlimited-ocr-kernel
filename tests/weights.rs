use half::bf16;
use uocr::{
    checkpoint::{DType, WeightView},
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};

#[test]
fn mapped_bf16_linear_handles_odd_rows_and_column_tails() {
    for rows in [37, 1025] {
        let cols = 131;
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 29) as f32 - 14.) / 16.)
            .collect();
        let bytes: Vec<u8> = w
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect();
        let shape = [rows, cols];
        let view = WeightView {
            dtype: DType::BF16,
            shape: &shape,
            bytes: &bytes,
        };
        for batch in [1, 3] {
            let x = Tensor::new(
                vec![batch, cols],
                (0..batch * cols)
                    .map(|i| ((i % 13) as f32 - 6.) / 8.)
                    .collect(),
            )
            .unwrap();
            let weight = Tensor::new(shape.to_vec(), w.clone()).unwrap();
            let mut reference = CpuContext::new(Backend::Scalar, 1, 1 << 20).unwrap();
            let expected = ops::linear(&x, &weight, None, &mut reference).unwrap();
            let mut backends = vec![Backend::Scalar, Backend::Auto];
            if cfg!(target_arch = "aarch64") {
                backends.push(Backend::Neon);
            }
            for backend in backends {
                let mut cpu = CpuContext::new(backend, 3, 1 << 20).unwrap();
                let actual = ops::linear_weight(&x, view, None, &mut cpu).unwrap();
                assert_eq!(actual.shape, expected.shape);
                for (a, b) in actual.data.iter().zip(&expected.data) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
#[cfg(target_arch = "aarch64")]
fn neon_dot_matches_f64_oracle_with_unaligned_storage_and_every_tail() {
    let mut cpu = CpuContext::new(Backend::Neon, 1, 1 << 20).unwrap();
    for cols in (1..=47).chain([128, 896, 1280, 6848]) {
        let weights: Vec<f32> = (0..cols)
            .map(|i| bf16::from_f32((i as f32 * 0.73).sin()).to_f32())
            .collect();
        let x = Tensor::new(
            vec![1, cols],
            (0..cols).map(|i| (i as f32 * 0.31).cos()).collect(),
        )
        .unwrap();
        let mut bytes = vec![0u8];
        bytes.extend(
            weights
                .iter()
                .flat_map(|w| bf16::from_f32(*w).to_bits().to_le_bytes()),
        );
        let shape = [1, cols];
        let view = WeightView {
            dtype: DType::BF16,
            shape: &shape,
            bytes: &bytes[1..],
        };
        let expected: f64 = weights
            .iter()
            .zip(&x.data)
            .map(|(&w, &x)| w as f64 * x as f64)
            .sum::<f64>()
            + 0.125;
        let actual = ops::linear_weight(&x, view, Some(&[0.125]), &mut cpu).unwrap();
        assert!(
            (actual.data[0] as f64 - expected).abs() < 2e-4,
            "cols={cols}: {} vs {expected}",
            actual.data[0]
        );
    }
}
