use crate::error::{OcrError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub ngram_size: usize,
    pub ngram_window: usize,
}
impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            ngram_size: 35,
            ngram_window: 128,
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Eos,
    TokenLimit,
}

pub fn select_token(history: &[u32], logits: &[f32], options: &GenerationOptions) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
        return Err(OcrError::Numerical("invalid generation logits".into()));
    }
    let mut banned = Vec::new();
    let n = options.ngram_size;
    if n > 0 && options.ngram_window > 0 && history.len() >= n {
        let start = history.len().saturating_sub(options.ngram_window);
        let end = history.len() - n + 1;
        for i in start..end {
            if n == 1 || history[i..i + n - 1] == history[history.len() - n + 1..] {
                banned.push(history[i + n - 1]);
            }
        }
    }
    let mut best = None;
    let mut score = f32::NEG_INFINITY;
    for (i, &value) in logits.iter().enumerate() {
        if value > score && !banned.contains(&(i as u32)) {
            score = value;
            best = Some(i as u32);
        }
    }
    best.ok_or_else(|| OcrError::Numerical("all generation tokens masked".into()))
}
