use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use uocr::{
    checkpoint::Checkpoint,
    cpu::{Backend, CpuContext, quantized::QuantizedWeights},
    tensor::Tensor,
};
fn write(path: &Path, header: serde_json::Value, data: &[u8]) {
    let json = serde_json::to_vec(&header).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(data);
    fs::write(path, bytes).unwrap();
}
fn fixture(dir: &Path) -> (Checkpoint, std::path::PathBuf) {
    let source = dir.join("source.safetensors");
    write(
        &source,
        serde_json::json!({"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}),
        &[1f32.to_le_bytes(), (-2f32).to_le_bytes()].concat(),
    );
    let hash = format!("{:x}", Sha256::digest(fs::read(&source).unwrap()));
    let path = dir.join("q.safetensors");
    let mut data = vec![1u8, 254];
    data.extend(1f32.to_le_bytes());
    write(
        &path,
        serde_json::json!({"__metadata__":{"format":"uocr-q8-block32-v1","block_size":"32","source_sha256":hash},"lm_head.weight":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]},"lm_head.weight.q8_scales":{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}),
        &data,
    );
    let bytes = fs::read(&path).unwrap();
    fs::write(path.with_extension("json"),serde_json::to_vec(&serde_json::json!({"schema_version":1,"format":"uocr-q8-block32-v1","block_size":32,"source_sha256":hash,"output_sha256":format!("{:x}",Sha256::digest(&bytes)),"matrix_count":1,"bytes":bytes.len()})).unwrap()).unwrap();
    (Checkpoint::open(&source).unwrap(), path)
}
#[test]
fn validated_overlay_computes_and_rejects_corruption_or_wrong_source() {
    let dir = tempfile::tempdir().unwrap();
    let (source, path) = fixture(dir.path());
    let q = QuantizedWeights::load(&path, &source).unwrap();
    let mut cpu = CpuContext::new(Backend::Auto, 1, 1024).unwrap();
    assert_eq!(
        q.get("lm_head.weight")
            .unwrap()
            .linear(&Tensor::new(vec![1, 2], vec![3., 4.]).unwrap(), &mut cpu)
            .unwrap()
            .data,
        [-5.]
    );
    assert!(q.get("model.norm.weight").is_none());
    let original = fs::read(&path).unwrap();
    let mut bytes = original.clone();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&path, bytes).unwrap();
    assert!(QuantizedWeights::load(&path, &source).is_err());
    fs::write(&path, original).unwrap();
    let wrong = dir.path().join("other.safetensors");
    write(
        &wrong,
        serde_json::json!({"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}),
        &[0; 8],
    );
    assert!(QuantizedWeights::load(&path, &Checkpoint::open(&wrong).unwrap()).is_err());
}

#[test]
fn authenticated_but_invalid_overlay_is_rejected() {
    for invalid in ["version", "scale", "shape", "dtype", "range", "missing"] {
        let dir = tempfile::tempdir().unwrap();
        let (source, path) = fixture(dir.path());
        let bytes = fs::read(&path).unwrap();
        let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
        let mut data = bytes[8 + len..].to_vec();
        match invalid {
            "version" => header["__metadata__"]["format"] = "future".into(),
            "scale" => data[2..6].copy_from_slice(&f32::NAN.to_le_bytes()),
            "shape" => header["lm_head.weight"]["shape"] = serde_json::json!([2, 1]),
            "dtype" => header["lm_head.weight"]["dtype"] = "F16".into(),
            "range" => data[0] = 128,
            "missing" => {
                header
                    .as_object_mut()
                    .unwrap()
                    .remove("lm_head.weight.q8_scales");
            }
            _ => unreachable!(),
        }
        write(&path, header, &data);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(path.with_extension("json")).unwrap()).unwrap();
        let bytes = fs::read(&path).unwrap();
        manifest["output_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
        manifest["bytes"] = bytes.len().into();
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        assert!(
            QuantizedWeights::load(&path, &source).is_err(),
            "accepted {invalid}"
        );
    }
}
