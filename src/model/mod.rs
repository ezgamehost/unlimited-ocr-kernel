pub mod config;
use crate::{
    checkpoint::Checkpoint,
    cpu::CpuContext,
    decoder::Decoder,
    error::{OcrError, Result},
    generation::{FinishReason, GenerationOptions, select_token},
    preprocess::{self, ImageMode},
    prompt::PromptEncoder,
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
        match (&self.quantized, self.quantization_mode) {
            (None, _) => "bf16_weights_fp32",
            (Some(_), crate::cpu::quantized::QuantizationMode::WeightsOnly) => "int8_weights_fp32",
            (Some(_), crate::cpu::quantized::QuantizationMode::DynamicActivations) => {
                "int8_weights_dynamic_activations"
            }
        }
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
        let image_tokens = image.image_tokens();
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
        for (row, &position) in input.image_positions.iter().enumerate() {
            let width = self.config.hidden_size;
            embeddings.data[position * width..(position + 1) * width]
                .copy_from_slice(&features.data[row * width..(row + 1) * width]);
        }
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
