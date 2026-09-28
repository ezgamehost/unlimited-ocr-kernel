use crate::error::{OcrError, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct PromptEncoder {
    tokenizer: Tokenizer,
}
pub struct PromptInput {
    pub ids: Vec<u32>,
    pub image_positions: Vec<usize>,
}
impl PromptEncoder {
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| OcrError::Format(e.to_string()))?;
        Ok(Self { tokenizer })
    }
    pub fn encode(&self, prompt: &str, image_tokens: usize) -> Result<PromptInput> {
        let pieces: Vec<_> = prompt.trim().split("<image>").collect();
        if pieces.len() != 2 || image_tokens == 0 {
            return Err(OcrError::Shape(
                "prompt must contain exactly one image marker".into(),
            ));
        }
        let before = self
            .tokenizer
            .encode(pieces[0], false)
            .map_err(|e| OcrError::Format(e.to_string()))?;
        let after = self
            .tokenizer
            .encode(pieces[1], false)
            .map_err(|e| OcrError::Format(e.to_string()))?;
        let mut ids = vec![0];
        ids.extend_from_slice(before.get_ids());
        let start = ids.len();
        ids.extend(std::iter::repeat_n(128815, image_tokens));
        ids.extend_from_slice(after.get_ids());
        Ok(PromptInput {
            ids,
            image_positions: (start..start + image_tokens).collect(),
        })
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| OcrError::Format(e.to_string()))
    }
}
