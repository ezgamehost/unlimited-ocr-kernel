use std::io::Write;
use tempfile::NamedTempFile;
use uocr::checkpoint::Checkpoint;

fn fixture(header: &str, data: &[u8]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(data).unwrap();
    file
}

#[test]
fn reads_bf16_and_f32_without_expanding_checkpoint() {
    let file = fixture(
        r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        &[0x80, 0x3f, 0, 0xc0, 0, 0, 0x40, 0x40],
    );
    let model = Checkpoint::open(file.path()).unwrap();
    assert_eq!(model.tensor("x").unwrap().to_f32(), vec![1.0, -2.0]);
    assert_eq!(model.tensor("y").unwrap().to_f32(), vec![3.0]);
    assert!(model.tensor("missing").is_err());
}

#[test]
fn rejects_truncation_overlap_and_overflow() {
    for header in [
        r#"{"x":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        r#"{"x":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,4]}}"#,
    ] {
        let file = fixture(header, &[0; 4]);
        assert!(Checkpoint::open(file.path()).is_err(), "accepted {header}");
    }
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&u64::MAX.to_le_bytes()).unwrap();
    assert!(Checkpoint::open(file.path()).is_err());
}

#[test]
fn rejects_shape_byte_count_mismatch_and_trailing_payload() {
    for header in [
        r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
    ] {
        let file = fixture(header, &[0; 4]);
        assert!(Checkpoint::open(file.path()).is_err());
    }
}
