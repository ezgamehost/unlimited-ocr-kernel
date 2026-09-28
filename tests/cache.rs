use uocr::decoder::cache::LayerCache;

#[test]
fn head_major_preserves_prefix_active_count_and_ring_slots() {
    let (heads, dim, prefix, window) = (3, 2, 5, 3);
    let width = heads * dim;
    let mut reference = LayerCache::new(heads, dim, window).unwrap();
    let mut packed = LayerCache::new(heads, dim, window).unwrap();
    assert!(packed.pack_heads().is_err());
    let keys: Vec<_> = (0..prefix * width).map(|i| i as f32).collect();
    let values: Vec<_> = keys.iter().map(|x| -x).collect();
    reference.prefill(&keys, &values, prefix).unwrap();
    packed.prefill(&keys, &values, prefix).unwrap();
    packed.pack_heads().unwrap();
    packed.pack_heads().unwrap();
    assert!(packed.is_head_major());
    for step in 0..12 {
        assert_eq!(packed.tokens(), reference.tokens());
        assert_eq!(packed.position(), reference.position());
        for (a, b) in [
            (packed.keys(), reference.keys()),
            (packed.values(), reference.values()),
        ] {
            for head in 0..heads {
                for token in 0..packed.tokens() {
                    assert_eq!(
                        &a[(head * (prefix + window) + token) * dim
                            ..(head * (prefix + window) + token + 1) * dim],
                        &b[token * width + head * dim..token * width + (head + 1) * dim]
                    );
                }
            }
        }
        let key: Vec<_> = (0..width)
            .map(|i| (100 + step * width + i) as f32)
            .collect();
        let value: Vec<_> = key.iter().map(|x| -x).collect();
        reference.push(&key, &value).unwrap();
        packed.push(&key, &value).unwrap();
    }
    let position = packed.position();
    assert!(packed.push(&[0.], &[0.]).is_err());
    assert_eq!(packed.position(), position);
}

#[test]
fn prefix_is_preserved_across_multiple_ring_wraps() {
    let mut cache = LayerCache::new(1, 1, 3).unwrap();
    cache.prefill(&[10., 20.], &[100., 200.], 2).unwrap();
    for i in 0..10 {
        cache.push(&[i as f32], &[(i * 10) as f32]).unwrap();
        assert_eq!(&cache.keys()[..2], &[10., 20.]);
        assert_eq!(&cache.values()[..2], &[100., 200.]);
        assert_eq!(cache.tokens(), 2 + (i + 1).min(3));
    }
    let mut tail = cache.keys()[2..].to_vec();
    tail.sort_by(f32::total_cmp);
    assert_eq!(tail, [7., 8., 9.]);
    assert_eq!(cache.position(), 12);
}

#[test]
fn bad_push_does_not_advance_cache() {
    let mut cache = LayerCache::new(2, 2, 128).unwrap();
    assert!(cache.push(&[1.; 4], &[1.; 4]).is_err());
    cache.prefill(&[0.; 8], &[0.; 8], 2).unwrap();
    assert!(cache.push(&[1.; 3], &[1.; 4]).is_err());
    assert_eq!(cache.position(), 2);
    assert!(cache.prefill(&[0.; 8], &[0.; 8], 2).is_err());
}
