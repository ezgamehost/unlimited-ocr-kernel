use image::RgbImage;
use std::path::Path;
use uocr::preprocess::{ImageMode, PreparedImage, prepare, resize_rgb};
use uocr::tensor::Tensor;

#[test]
fn prepared_images_reject_invalid_grids_and_storage() {
    let mut image = PreparedImage {
        global: Tensor::new(vec![1, 1024, 1024, 3], vec![0.; 1024 * 1024 * 3]).unwrap(),
        crops: vec![],
        grid: [1, 1],
    };
    assert!(image.validate().is_ok());
    assert_eq!(image.image_tokens().unwrap(), 273);
    for grid in [[0, 1], [1, 0], [usize::MAX, 2], [1, 33], [2, 2]] {
        image.grid = grid;
        assert!(image.validate().is_err(), "grid={grid:?}");
        assert!(image.image_tokens().is_err());
    }
    image.grid = [2, 1];
    image.crops = vec![
        Tensor {
            shape: vec![1, 640, 640, 3],
            data: vec![]
        };
        2
    ];
    assert!(image.validate().is_err());
    for crop in &mut image.crops {
        crop.data = vec![0.; 640 * 640 * 3];
    }
    assert!(image.validate().is_ok());
    assert_eq!(image.image_tokens().unwrap(), 483);
    image.global.data.pop();
    assert!(image.validate().is_err());
}

#[test]
fn bicubic_matches_pillow_fixed_point_fixture() {
    let input = RgbImage::from_raw(
        3,
        2,
        vec![
            0, 10, 20, 100, 110, 120, 200, 210, 220, 255, 245, 235, 155, 145, 135, 55, 45, 35,
        ],
    )
    .unwrap();
    let output = resize_rgb(&input, 2, 3).unwrap();
    let expected: Vec<u8> = serde_json::from_str(include_str!("fixtures/resize_rgb.json")).unwrap();
    assert_eq!(output.as_raw(), &expected);
}

#[test]
#[ignore = "requires original model fixtures"]
fn normalized_logo_pixels_match_original_exactly() {
    let image = prepare(
        Path::new("/tmp/unlimited-ocr-baseline/baidu.png"),
        ImageMode::Base,
    )
    .unwrap();
    let reference = uocr::checkpoint::Checkpoint::open(Path::new(
        "artifacts/reference-logo-f32/preprocess.safetensors",
    ))
    .unwrap();
    let expected = reference.tensor("global").unwrap().to_f32();
    assert_eq!(image.global.shape, [1, 1024, 1024, 3]);
    for y in 0..1024 {
        for x in 0..1024 {
            for c in 0..3 {
                assert_eq!(
                    image.global.data[(y * 1024 + x) * 3 + c],
                    expected[c * 1024 * 1024 + y * 1024 + x]
                );
            }
        }
    }
}
