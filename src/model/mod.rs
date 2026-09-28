#[cfg(test)]
mod benchmark;
pub mod config;
use crate::{
    checkpoint::Checkpoint,
    cpu::CpuContext,
    decoder::Decoder,
    error::{OcrError, Result},
    generation::{FinishReason, GenerationOptions, select_token},
    preprocess::{self, ImageMode},
    prompt::PromptEncoder,
    tensor::Tensor,
    vision::Vision,
};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn default_prompt() -> String {
    "<image>document parsing.".into()
}

fn scatter_image_features(
    embeddings: &mut Tensor,
    features: &Tensor,
    positions: &[usize],
) -> Result<()> {
    embeddings.validate()?;
    features.validate()?;
    if embeddings.shape.len() != 2
        || embeddings.shape[1] == 0
        || features.shape != [positions.len(), embeddings.shape[1]]
        || positions
            .iter()
            .any(|&position| position >= embeddings.shape[0])
    {
        return Err(OcrError::Shape(
            "image features and prompt embeddings disagree".into(),
        ));
    }
    let width = embeddings.shape[1];
    for (row, &position) in positions.iter().enumerate() {
        embeddings.data[position * width..(position + 1) * width]
            .copy_from_slice(&features.data[row * width..(row + 1) * width]);
    }
    Ok(())
}

fn default_mode() -> ImageMode {
    ImageMode::Gundam
}
#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub image: PathBuf,
    #[serde(default = "default_mode")]
    pub mode: ImageMode,
    #[serde(default = "default_prompt")]
    pub prompt: String,
    #[serde(default)]
    pub generation: GenerationOptions,
}
#[derive(Debug, Serialize)]
pub struct Timings {
    pub preprocess_seconds: f64,
    pub vision_seconds: f64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub first_token_seconds: f64,
    pub total_seconds: f64,
    pub decode_steps: usize,
}
#[derive(Debug, Serialize)]
pub struct OcrResult {
    pub precision: &'static str,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub timings: Timings,
    pub image_tokens: usize,
}
pub struct Model {
    quantized: Option<crate::cpu::quantized::QuantizedWeights>,
    quantization_mode: crate::cpu::quantized::QuantizationMode,
    checkpoint: Checkpoint,
    config: config::ModelConfig,
    vision: Vision,
    tokenizer: PromptEncoder,
}

impl Model {
    pub fn precision(&self) -> &'static str {
        self.quantized
            .as_ref()
            .map_or("bf16_weights_fp32", |q| q.precision(self.quantization_mode))
    }
    pub fn load(directory: &Path) -> Result<Self> {
        Self::load_with_quantization(
            directory,
            None,
            crate::cpu::quantized::QuantizationMode::WeightsOnly,
        )
    }

    pub fn load_with_quantization(
        directory: &Path,
        overlay: Option<&Path>,
        mode: crate::cpu::quantized::QuantizationMode,
    ) -> Result<Self> {
        if overlay.is_none() && mode == crate::cpu::quantized::QuantizationMode::DynamicActivations
        {
            return Err(OcrError::Format(
                "dynamic INT8 requires a quantized weight overlay".into(),
            ));
        }
        let config = config::ModelConfig::load(&directory.join("config.json"))?;
        if config.hidden_size != 1280 {
            return Err(OcrError::Unsupported(
                "vision projector requires decoder hidden_size 1280".into(),
            ));
        }
        let checkpoint = Checkpoint::open(&directory.join("model-00001-of-000001.safetensors"))?;
        Decoder::new(&checkpoint, config.clone())?;
        let quantized = overlay
            .map(|path| crate::cpu::quantized::QuantizedWeights::load(path, &checkpoint))
            .transpose()?;
        let vision = Vision::new(&checkpoint)?;
        let tokenizer = PromptEncoder::load(&directory.join("tokenizer.json"))?;
        Ok(Self {
            quantized,
            quantization_mode: mode,
            checkpoint,
            config,
            vision,
            tokenizer,
        })
    }

    pub fn infer(&self, request: &Request, cpu: &mut CpuContext) -> Result<OcrResult> {
        if request.generation.max_tokens == 0 {
            return Err(OcrError::Shape("max_tokens must be positive".into()));
        }
        let start = Instant::now();
        let image = preprocess::prepare(&request.image, request.mode)?;
        let image_tokens = image.image_tokens()?;
        let input = self.tokenizer.encode(&request.prompt, image_tokens)?;
        if input
            .ids
            .len()
            .checked_add(request.generation.max_tokens)
            .is_none_or(|n| n > self.config.max_position_embeddings)
        {
            return Err(OcrError::Resource(
                "prompt and output cap exceed context limit".into(),
            ));
        }
        let preprocess_seconds = start.elapsed().as_secs_f64();
        let stamp = Instant::now();
        let features = self.vision.encode(&image, cpu)?;
        let vision_seconds = stamp.elapsed().as_secs_f64();
        let stamp = Instant::now();
        let mut decoder = Decoder::new(&self.checkpoint, self.config.clone())?;
        if let Some(quantized) = &self.quantized {
            decoder = decoder.with_quantization(quantized, self.quantization_mode);
        }
        let mut embeddings = decoder.embedding(&input.ids)?;
        scatter_image_features(&mut embeddings, &features, &input.image_positions)?;
        let (mut state, mut logits) = decoder.prefill(&embeddings, cpu)?;
        let prefill_seconds = stamp.elapsed().as_secs_f64();
        let mut history = input.ids;
        let mut output = Vec::new();
        let mut reason = FinishReason::TokenLimit;
        let mut first_token_seconds = 0.;
        let mut decode_seconds = 0.;
        let mut decode_steps = 0;
        for step in 0..request.generation.max_tokens {
            let token = select_token(&history, &logits.data, &request.generation)?;
            if step == 0 {
                first_token_seconds = start.elapsed().as_secs_f64();
            }
            if token == 1 {
                reason = FinishReason::Eos;
                break;
            }
            output.push(token);
            history.push(token);
            if step + 1 == request.generation.max_tokens {
                break;
            }
            let stamp = Instant::now();
            logits = decoder.step(token, &mut state, cpu)?;
            decode_seconds += stamp.elapsed().as_secs_f64();
            decode_steps += 1;
        }
        let text = self.tokenizer.decode(&output)?;
        let timings = Timings {
            preprocess_seconds,
            vision_seconds,
            prefill_seconds,
            decode_seconds,
            first_token_seconds,
            total_seconds: start.elapsed().as_secs_f64(),
            decode_steps,
        };
        Ok(OcrResult {
            precision: self.precision(),
            text,
            token_ids: output,
            finish_reason: reason,
            timings,
            image_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn image_scatter_validates_shape_and_positions_before_mutation() {
        let original = Tensor::new(vec![4, 3], vec![-1.; 12]).unwrap();
        let features = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let mut embeddings = original.clone();
        scatter_image_features(&mut embeddings, &features, &[1, 3]).unwrap();
        assert_eq!(
            embeddings.data,
            [-1., -1., -1., 1., 2., 3., -1., -1., -1., 4., 5., 6.]
        );
        for shape in [vec![3, 2], vec![1, 6], vec![1, 2, 3]] {
            let mut invalid = features.clone();
            invalid.shape = shape;
            let mut embeddings = original.clone();
            assert!(scatter_image_features(&mut embeddings, &invalid, &[1, 3]).is_err());
            assert_eq!(embeddings.data, original.data);
        }
        for positions in [vec![1], vec![1, 4], vec![1, usize::MAX]] {
            let mut embeddings = original.clone();
            assert!(scatter_image_features(&mut embeddings, &features, &positions).is_err());
            assert_eq!(embeddings.data, original.data);
        }
        let mut invalid = features.clone();
        invalid.data.pop();
        let mut embeddings = original.clone();
        assert!(scatter_image_features(&mut embeddings, &invalid, &[1, 3]).is_err());
        assert_eq!(embeddings.data, original.data);
    }
}
