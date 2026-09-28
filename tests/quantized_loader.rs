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

#[test]
fn packed_overlay_dispatch_and_authenticated_validation() {
    for invalid in [
        "valid",
        "version",
        "weight_bits",
        "storage_bits",
        "header_bits",
        "padding",
        "code",
        "dtype",
        "shape",
        "scale",
        "source",
        "checksum",
        "missing",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (source, path) = fixture(dir.path());
        let hash = source.sha256().unwrap();
        let mut header = serde_json::json!({"__metadata__":{"format":"uocr-q4-block32-v1","block_size":"32","source_sha256":hash,"weight_bits":"4","storage_bits":"4"},"lm_head.weight":{"dtype":"U8","shape":[1,1,16],"data_offsets":[0,16]},"lm_head.weight.q4_scales":{"dtype":"F32","shape":[1,1],"data_offsets":[16,20]}});
        let mut data = vec![0u8; 16];
        data[0] = 1;
        data[1] = 14;
        data.extend(1f32.to_le_bytes());
        match invalid {
            "version" => header["__metadata__"]["format"] = "future".into(),
            "header_bits" => header["__metadata__"]["storage_bits"] = "8".into(),
            "padding" => data[0] |= 16,
            "code" => data[0] = 8,
            "dtype" => header["lm_head.weight"]["dtype"] = "I8".into(),
            "shape" => header["lm_head.weight"]["shape"] = serde_json::json!([1, 16]),
            "scale" => data[16..].copy_from_slice(&f32::NAN.to_le_bytes()),
            "missing" => {
                header
                    .as_object_mut()
                    .unwrap()
                    .remove("lm_head.weight.q4_scales");
            }
            _ => {}
        }
        write(&path, header, &data);
        let bytes = fs::read(&path).unwrap();
        let mut manifest = serde_json::json!({"schema_version":1,"format":"uocr-q4-block32-v1","block_size":32,"source_sha256":hash,"output_sha256":format!("{:x}",Sha256::digest(&bytes)),"matrix_count":1,"bytes":bytes.len(),"weight_bits":4,"storage_bits":4});
        match invalid {
            "weight_bits" | "storage_bits" => {
                manifest.as_object_mut().unwrap().remove(invalid);
            }
            "source" => manifest["source_sha256"] = "wrong".into(),
            "checksum" => manifest["output_sha256"] = "wrong".into(),
            _ => {}
        }
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let result = QuantizedWeights::load(&path, &source);
        if invalid == "valid" {
            let overlay = result.unwrap();
            assert_eq!(
                overlay.precision(uocr::cpu::quantized::QuantizationMode::WeightsOnly),
                "int4_weights_fp32"
            );
            assert_eq!(
                overlay.precision(uocr::cpu::quantized::QuantizationMode::DynamicActivations),
                "int4_weights_dynamic_int8_activations"
            );
            let mut cpu = CpuContext::new(Backend::Auto, 1, 1024).unwrap();
            let matrix = overlay.get("lm_head.weight").unwrap();
            let x = Tensor::new(vec![1, 2], vec![3., 4.]).unwrap();
            assert_eq!(matrix.linear(&x, &mut cpu).unwrap().data, [-5.]);
            assert!((matrix.linear_dynamic_int8(&x, &mut cpu).unwrap().data[0] + 5.).abs() < 0.02);
        } else {
            assert!(result.is_err(), "accepted {invalid}");
        }
    }
}

#[test]
fn hybrid_overlay_requires_q8_head_and_q4_body() {
    use uocr::cpu::quantized::QuantizationMode;
    let body = "model.layers.0.self_attn.q_proj.weight";
    for invalid in [
        "valid",
        "body_bits",
        "head_bits",
        "header_bits",
        "head_dtype",
        "body_dtype",
        "head_range",
        "body_padding",
        "head_suffix",
        "missing_head",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.safetensors");
        let mut source_header = serde_json::json!({"lm_head.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},body:{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]}});
        let source_data = [1f32.to_le_bytes(), (-2f32).to_le_bytes()].concat();
        if invalid == "missing_head" {
            source_header
                .as_object_mut()
                .unwrap()
                .remove("lm_head.weight");
            source_header[body]["data_offsets"] = serde_json::json!([0, 8]);
            write(&source_path, source_header, &source_data);
        } else {
            write(
                &source_path,
                source_header,
                &[source_data.clone(), source_data].concat(),
            );
        }
        let source = Checkpoint::open(&source_path).unwrap();
        let hash = source.sha256().unwrap();
        let mut header = serde_json::json!({"__metadata__":{"format":"uocr-q4-q8head-block32-v1","block_size":"32","source_sha256":hash,"body_bits":"4","head_bits":"8"},"lm_head.weight":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]},"lm_head.weight.q8_scales":{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]},body:{"dtype":"U8","shape":[1,1,16],"data_offsets":[6,22]},format!("{body}.q4_scales"):{"dtype":"F32","shape":[1,1],"data_offsets":[22,26]}});
        let mut data = vec![1, 254];
        data.extend(1f32.to_le_bytes());
        data.extend([1, 14]);
        data.extend([0; 14]);
        data.extend(1f32.to_le_bytes());
        match invalid {
            "header_bits" => header["__metadata__"]["head_bits"] = "4".into(),
            "head_dtype" => header["lm_head.weight"]["dtype"] = "U8".into(),
            "body_dtype" => header[body]["dtype"] = "I8".into(),
            "head_range" => data[0] = 128,
            "body_padding" => data[6] = 17,
            "head_suffix" => {
                let value = header
                    .as_object_mut()
                    .unwrap()
                    .remove("lm_head.weight.q8_scales")
                    .unwrap();
                header["lm_head.weight.q4_scales"] = value;
            }
            "missing_head" => {
                header.as_object_mut().unwrap().remove("lm_head.weight");
                header
                    .as_object_mut()
                    .unwrap()
                    .remove("lm_head.weight.q8_scales");
                header[body]["data_offsets"] = serde_json::json!([0, 16]);
                header[format!("{body}.q4_scales")]["data_offsets"] = serde_json::json!([16, 20]);
                data = data[6..].to_vec();
            }
            _ => {}
        }
        let path = dir.path().join("hybrid.safetensors");
        write(&path, header, &data);
        let bytes = fs::read(&path).unwrap();
        let mut manifest = serde_json::json!({"schema_version":1,"format":"uocr-q4-q8head-block32-v1","block_size":32,"source_sha256":hash,"output_sha256":format!("{:x}",Sha256::digest(&bytes)),"matrix_count":if invalid=="missing_head" {1} else {2},"bytes":bytes.len(),"body_bits":4,"head_bits":8});
        if invalid == "body_bits" || invalid == "head_bits" {
            manifest.as_object_mut().unwrap().remove(invalid);
        }
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let result = QuantizedWeights::load(&path, &source);
        if invalid == "valid" {
            let weights = result.unwrap();
            assert_eq!(
                weights.precision(QuantizationMode::DynamicActivations),
                "int4_body_int8_head_dynamic_int8_activations"
            );
            assert_eq!(
                weights.precision(QuantizationMode::WeightsOnly),
                "int4_body_int8_head_fp32"
            );
            let mut cpu = CpuContext::new(Backend::Auto, 1, 1024).unwrap();
            let x = Tensor::new(vec![1, 2], vec![3., 4.]).unwrap();
            for name in ["lm_head.weight", body] {
                assert_eq!(
                    weights
                        .get(name)
                        .unwrap()
                        .linear(&x, &mut cpu)
                        .unwrap()
                        .data,
                    [-5.]
                );
            }
        } else {
            assert!(result.is_err(), "accepted {invalid}");
        }
    }
}
