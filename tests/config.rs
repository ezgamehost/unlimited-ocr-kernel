use uocr::model::config::ModelConfig;

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
