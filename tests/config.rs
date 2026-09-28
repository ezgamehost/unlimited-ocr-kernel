use uocr::model::config::ModelConfig;

fn valid_config_json() -> serde_json::Value {
    serde_json::json!({"model_type":"unlimited-ocr","hidden_size":1280,
        "intermediate_size":6848,"moe_intermediate_size":896,"num_hidden_layers":12,
        "num_attention_heads":10,"num_key_value_heads":10,"n_routed_experts":64,
        "n_shared_experts":2,"num_experts_per_tok":6,"first_k_dense_replace":1,
        "use_mla":false,"vocab_size":129280})
}

#[test]
fn rejects_overflowing_derived_decoder_dimensions() {
    for field in ["n_shared_experts", "vocab_size", "max_position_embeddings"] {
        let mut json = valid_config_json();
        json[field] = serde_json::json!(usize::MAX);
        assert!(
            ModelConfig::from_json(&json.to_string()).is_err(),
            "{field}"
        );
    }
}

#[test]
fn decoder_validates_direct_config_before_checkpoint_descriptors() {
    use std::io::Write;
    use uocr::{checkpoint::Checkpoint, decoder::Decoder, error::OcrError};
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&[0; 4]).unwrap();
    let weights = Checkpoint::open(file.path()).unwrap();
    let valid = ModelConfig::from_json(&valid_config_json().to_string()).unwrap();
    let mut config = valid.clone();
    config.num_attention_heads = 0;
    assert!(matches!(
        Decoder::new(&weights, config),
        Err(OcrError::Shape(_))
    ));
    let mut config = valid.clone();
    config.n_shared_experts = usize::MAX;
    assert!(matches!(
        Decoder::new(&weights, config),
        Err(OcrError::Shape(_))
    ));
    let mut config = valid;
    config.use_mla = true;
    assert!(matches!(
        Decoder::new(&weights, config),
        Err(OcrError::Unsupported(_))
    ));
}

#[test]
fn resolves_upstream_decoder_defaults() {
    let json = r#"{"model_type":"unlimited-ocr","hidden_size":1280,"intermediate_size":6848,"moe_intermediate_size":896,"num_hidden_layers":12,"num_attention_heads":10,"num_key_value_heads":10,"n_routed_experts":64,"n_shared_experts":2,"num_experts_per_tok":6,"first_k_dense_replace":1,"use_mla":false,"vocab_size":129280,"sliding_window":128}"#;
    let cfg = ModelConfig::from_json(json).unwrap();
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 10000.);
    assert!(!cfg.norm_topk_prob);
    assert_eq!(cfg.routed_scaling_factor, 1.);
    assert!(
        ModelConfig::from_json(&json.replace("\"use_mla\":false", "\"use_mla\":true")).is_err()
    );
    assert!(
        ModelConfig::from_json(
            &json.replace("\"num_attention_heads\":10", "\"num_attention_heads\":9")
        )
        .is_err()
    );
}
