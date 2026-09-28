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
fn reads_unsigned_packed_bytes_without_treating_them_as_linear_weights() {
    use uocr::{
        cpu::{Backend, CpuContext, ops},
        tensor::Tensor,
    };
    let file = fixture(
        r#"{"x":{"dtype":"U8","shape":[1,2],"data_offsets":[0,2]}}"#,
        &[0x79, 0xe3],
    );
    let model = Checkpoint::open(file.path()).unwrap();
    let view = model.tensor("x").unwrap();
    assert_eq!(view.bytes, &[0x79, 0xe3]);
    assert_eq!(view.to_f32(), vec![121., 227.]);
    let input = Tensor::new(vec![1, 2], vec![1., 1.]).unwrap();
    let mut cpu = CpuContext::new(Backend::Scalar, 1, 1024).unwrap();
    assert!(ops::linear_weight(&input, view, None, &mut cpu).is_err());
    let truncated = fixture(
        r#"{"x":{"dtype":"U8","shape":[1,3],"data_offsets":[0,3]}}"#,
        &[0x79, 0xe3],
    );
    assert!(Checkpoint::open(truncated.path()).is_err());
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

#[test]
fn checksum_streams_multiple_chunks_repeats_and_uses_original_file() {
    use sha2::{Digest, Sha256};
    let data: Vec<u8> = (0..3 * 1024 * 1024 + 19).map(|i| (i * 113) as u8).collect();
    let header = format!(
        r#"{{"x":{{"dtype":"I8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        data.len(),
        data.len()
    );
    let fixture = fixture(&header, &data);
    let bytes = std::fs::read(fixture.path()).unwrap();
    let expected = format!("{:x}", Sha256::digest(&bytes));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    std::fs::write(&path, bytes).unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    // Atomic pathname replacement leaves the mapped inode and retained file intact.
    std::fs::rename(&path, dir.path().join("original.safetensors")).unwrap();
    std::fs::write(&path, b"a different file").unwrap();
    assert_eq!(checkpoint.sha256().unwrap(), expected);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| checkpoint.sha256().unwrap());
        let second = scope.spawn(|| checkpoint.sha256().unwrap());
        assert_eq!(first.join().unwrap(), expected);
        assert_eq!(second.join().unwrap(), expected);
    });
}
