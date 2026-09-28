use uocr::{
    cpu::{Backend, CpuContext, ops},
    tensor::Tensor,
};

#[test]
fn head_major_decode_preserves_bits_and_ignores_unused_capacity() {
    let (heads, dim) = (10, 128);
    let width = heads * dim;
    for (tokens, capacity) in [(1, 129), (129, 257), (1518, 1646), (1646, 1646)] {
        let q = Tensor::new(
            vec![1, width],
            (0..width).map(|i| (i as f32 * 0.17).sin()).collect(),
        )
        .unwrap();
        let keys: Vec<_> = (0..tokens * width)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();
        let values: Vec<_> = (0..tokens * width)
            .map(|i| (i as f32 * 0.031).sin())
            .collect();
        let pack = |input: &[f32]| {
            let mut output = vec![f32::NAN; capacity * width];
            for head in 0..heads {
                for token in 0..tokens {
                    output[(head * capacity + token) * dim..(head * capacity + token + 1) * dim]
                        .copy_from_slice(
                            &input[token * width + head * dim..token * width + (head + 1) * dim],
                        );
                }
            }
            output
        };
        let (pk, pv) = (pack(&keys), pack(&values));
        for backend in [Backend::Scalar, Backend::Auto] {
            for (threads, scratch) in [
                (1, (tokens + width) * 4),
                (8, (tokens * heads + width) * 4),
                (8, (tokens + width) * 4),
            ] {
                let mut cpu = CpuContext::new(backend, threads, scratch).unwrap();
                let a = ops::attention_decode(&q, &keys, &values, heads, &mut cpu).unwrap();
                let b = ops::attention_decode_head_major(&q, &pk, &pv, heads, tokens, &mut cpu)
                    .unwrap();
                assert_eq!(
                    a.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
                assert!(
                    ops::attention_decode_head_major(&q, &pk, &pv, heads, 0, &mut cpu).is_err()
                );
                assert!(
                    ops::attention_decode_head_major(&q, &pk, &pv, heads, capacity + 1, &mut cpu)
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn strided_decode_attention_matches_f64_oracle() {
    for tokens in [1, 7, 139, 1641] {
        let (heads, dim) = (3, 5);
        let width = heads * dim;
        let q = Tensor::new(
            vec![1, width],
            (0..width).map(|i| (i as f32 * 0.7).sin()).collect(),
        )
        .unwrap();
        let k: Vec<f32> = (0..tokens * width)
            .map(|i| (i as f32 * 0.13).cos())
            .collect();
        let v: Vec<f32> = (0..tokens * width)
            .map(|i| (i as f32 * 0.31).sin())
            .collect();
        let mut expected = vec![0.; width];
        for h in 0..heads {
            let scores: Vec<f64> = (0..tokens)
                .map(|t| {
                    ((0..dim)
                        .map(|d| q.data[h * dim + d] as f64 * k[t * width + h * dim + d] as f64)
                        .sum::<f64>()
                        / (dim as f64).sqrt())
                    .exp()
                })
                .collect();
            let sum: f64 = scores.iter().sum();
            for d in 0..dim {
                expected[h * dim + d] = (0..tokens)
                    .map(|t| scores[t] / sum * v[t * width + h * dim + d] as f64)
                    .sum::<f64>();
            }
        }
        for backend in [Backend::Scalar, Backend::Auto] {
            let mut cpu = CpuContext::new(backend, 2, (tokens + width) * 4).unwrap();
            let actual = ops::attention_decode(&q, &k, &v, heads, &mut cpu).unwrap();
            assert_eq!(actual.shape, q.shape);
            for (a, b) in actual.data.iter().zip(&expected) {
                assert!((*a as f64 - b).abs() < 2e-6, "tokens={tokens}, {a} vs {b}");
            }
        }
    }
}

#[test]
fn strided_decode_attention_rejects_invalid_layouts_and_budget() {
    let q = Tensor::new(vec![1, 6], vec![0.; 6]).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 1, 1024).unwrap();
    for (keys, values, heads) in [
        (vec![], vec![], 2),
        (vec![0.; 7], vec![0.; 7], 2),
        (vec![0.; 6], vec![0.; 5], 2),
        (vec![0.; 6], vec![0.; 6], 4),
        (vec![0.; 6], vec![0.; 6], 0),
    ] {
        assert!(ops::attention_decode(&q, &keys, &values, heads, &mut cpu).is_err());
    }
    let mut cpu = CpuContext::new(Backend::Auto, 1, 4).unwrap();
    assert!(ops::attention_decode(&q, &[0.; 6], &[0.; 6], 2, &mut cpu).is_err());
}

#[test]
fn decode_heads_preserve_bits_across_threads_and_scratch_fallback() {
    let (heads, dim) = (10, 128);
    let width = heads * dim;
    for tokens in [129, 1641] {
        let q = Tensor::new(
            vec![1, width],
            (0..width).map(|i| (i as f32 * 0.17).sin()).collect(),
        )
        .unwrap();
        let keys: Vec<f32> = (0..tokens * width)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();
        let values: Vec<f32> = (0..tokens * width)
            .map(|i| (i as f32 * 0.031).sin())
            .collect();
        let minimum = (tokens + width) * 4;
        let parallel = (tokens * heads + width) * 4;
        let mut results = Vec::new();
        for (threads, scratch) in [(1, minimum), (8, parallel), (8, minimum)] {
            let mut cpu = CpuContext::new(Backend::Auto, threads, scratch).unwrap();
            let output = ops::attention_decode(&q, &keys, &values, heads, &mut cpu).unwrap();
            results.push(output.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
        let mut cpu = CpuContext::new(Backend::Auto, 8, minimum - 1).unwrap();
        assert!(ops::attention_decode(&q, &keys, &values, heads, &mut cpu).is_err());
    }
}
