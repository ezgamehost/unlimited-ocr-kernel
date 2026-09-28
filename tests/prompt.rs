use std::path::Path;
use uocr::prompt::PromptEncoder;

#[test]
#[ignore = "requires pinned tokenizer"]
fn prompt_ids_equal_upstream_fixture() {
    let encoder = PromptEncoder::load(Path::new("artifacts/upstream/tokenizer.json")).unwrap();
    let input = encoder.encode("<image>document parsing.", 273).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("artifacts/reference-logo-f32/metadata.json").unwrap(),
    )
    .unwrap();
    let expected: Vec<u32> = serde_json::from_value(metadata["token_ids"].clone()).unwrap();
    assert_eq!(input.ids, expected);
    assert_eq!(input.image_positions, (1..274).collect::<Vec<_>>());
    assert!(encoder.encode("no image marker", 273).is_err());
}
