use uocr::decoder::cache::LayerCache;

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
