use uocr::decoder::moe::route;

#[test]
fn unnormalized_routing_keeps_probability_mass_of_unselected_experts() {
    let selected = route(&[0., 0., 0., 0.], 2, false, 1.).unwrap();
    assert_eq!(selected, vec![(0, 0.25), (1, 0.25)]);
    assert_eq!(
        route(&[0., 0., 0., 0.], 2, true, 2.).unwrap(),
        vec![(0, 1.), (1, 1.)]
    );
    assert!(route(&[0., f32::NAN], 1, false, 1.).is_err());
    assert!(route(&[0.], 2, false, 1.).is_err());
}
