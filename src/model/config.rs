use crate::error::{OcrError, Result};
use serde::Deserialize;
use std::path::Path;

fn eps() -> f32 {
    1e-6
}
fn theta() -> f32 {
    10000.
}
fn one() -> f32 {
    1.
}
fn window() -> usize {
    128
}
fn context() -> usize {
    32768
}
fn greedy() -> String {
    "greedy".into()
}
fn softmax() -> String {
    "softmax".into()
}
fn layer_freq() -> usize {
    1
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub use_mla: bool,
    pub vocab_size: usize,
    #[serde(default = "window")]
    pub sliding_window: usize,
    #[serde(default = "eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "one")]
    pub routed_scaling_factor: f32,
    #[serde(default = "context")]
    pub max_position_embeddings: usize,
    #[serde(default = "greedy")]
    pub topk_method: String,
    #[serde(default = "softmax")]
    pub scoring_func: String,
    #[serde(default = "layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub attention_bias: bool,
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        if config.model_type != "unlimited-ocr"
            || config.use_mla
            || config.attention_bias
            || config.rope_scaling.is_some()
            || config.topk_method != "greedy"
            || config.scoring_func != "softmax"
            || config.moe_layer_freq != 1
        {
            return Err(OcrError::Unsupported(
                "model attention or routing configuration".into(),
            ));
        }
        if config.hidden_size == 0
            || config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            || !(config.hidden_size / config.num_attention_heads).is_multiple_of(2)
            || config.num_attention_heads != config.num_key_value_heads
            || config.num_hidden_layers == 0
            || config.vocab_size == 0
            || config.intermediate_size == 0
            || config.moe_intermediate_size == 0
            || config.n_shared_experts == 0
            || config.num_experts_per_tok == 0
            || config.num_experts_per_tok > config.n_routed_experts
            || config.first_k_dense_replace > config.num_hidden_layers
            || config.sliding_window == 0
            || config.max_position_embeddings == 0
            || !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.
            || !config.rope_theta.is_finite()
            || config.rope_theta <= 0.
            || !config.routed_scaling_factor.is_finite()
            || config.routed_scaling_factor <= 0.
        {
            return Err(OcrError::Shape("invalid decoder configuration".into()));
        }
        Ok(config)
    }
}
