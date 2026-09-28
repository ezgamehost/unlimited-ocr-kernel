//! Consumed tensor descriptors for the pinned Unlimited-OCR vision architecture.
use crate::{
    checkpoint::{Checkpoint, DType},
    error::{OcrError, Result},
};
use std::collections::BTreeMap;

pub(super) fn validate(checkpoint: &Checkpoint) -> Result<Vec<String>> {
    let mut expected = BTreeMap::<String, Vec<usize>>::new();
    expected.insert("model.image_newline".into(), vec![1280]);
    expected.insert("model.projector.layers.bias".into(), vec![1280]);
    expected.insert("model.projector.layers.weight".into(), vec![1280, 2048]);
    expected.insert("model.sam_model.neck.0.weight".into(), vec![256, 768, 1, 1]);
    expected.insert("model.sam_model.neck.1.bias".into(), vec![256]);
    expected.insert("model.sam_model.neck.1.weight".into(), vec![256]);
    expected.insert("model.sam_model.neck.2.weight".into(), vec![256, 256, 3, 3]);
    expected.insert("model.sam_model.neck.3.bias".into(), vec![256]);
    expected.insert("model.sam_model.neck.3.weight".into(), vec![256]);
    expected.insert("model.sam_model.net_2.weight".into(), vec![512, 256, 3, 3]);
    expected.insert("model.sam_model.net_3.weight".into(), vec![1024, 512, 3, 3]);
    expected.insert("model.sam_model.patch_embed.proj.bias".into(), vec![768]);
    expected.insert(
        "model.sam_model.patch_embed.proj.weight".into(),
        vec![768, 3, 16, 16],
    );
    expected.insert("model.sam_model.pos_embed".into(), vec![1, 64, 64, 768]);
    expected.insert("model.view_seperator".into(), vec![1280]);
    expected.insert(
        "model.vision_model.embeddings.class_embedding".into(),
        vec![1024],
    );
    expected.insert(
        "model.vision_model.embeddings.position_embedding.weight".into(),
        vec![257, 1024],
    );
    expected.insert("model.vision_model.pre_layrnorm.bias".into(), vec![1024]);
    expected.insert("model.vision_model.pre_layrnorm.weight".into(), vec![1024]);
    for layer in 0..12 {
        let relative = if [2, 5, 8, 11].contains(&layer) {
            127
        } else {
            27
        };
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.proj.bias"),
            vec![768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.proj.weight"),
            vec![768, 768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.qkv.bias"),
            vec![2304],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.qkv.weight"),
            vec![2304, 768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.rel_pos_h"),
            vec![relative, 64],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.attn.rel_pos_w"),
            vec![relative, 64],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.mlp.lin1.bias"),
            vec![3072],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.mlp.lin1.weight"),
            vec![3072, 768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.mlp.lin2.bias"),
            vec![768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.mlp.lin2.weight"),
            vec![768, 3072],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.norm1.bias"),
            vec![768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.norm1.weight"),
            vec![768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.norm2.bias"),
            vec![768],
        );
        expected.insert(
            format!("model.sam_model.blocks.{layer}.norm2.weight"),
            vec![768],
        );
    }
    for layer in 0..24 {
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.layer_norm1.bias"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.layer_norm1.weight"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.layer_norm2.bias"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.layer_norm2.weight"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.mlp.fc1.bias"),
            vec![4096],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.mlp.fc1.weight"),
            vec![4096, 1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.mlp.fc2.bias"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.mlp.fc2.weight"),
            vec![1024, 4096],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.self_attn.out_proj.bias"),
            vec![1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.self_attn.out_proj.weight"),
            vec![1024, 1024],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.self_attn.qkv_proj.bias"),
            vec![3072],
        );
        expected.insert(
            format!("model.vision_model.transformer.layers.{layer}.self_attn.qkv_proj.weight"),
            vec![3072, 1024],
        );
    }
    let mut names = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        let tensor = checkpoint.tensor(&name)?;
        if !matches!(tensor.dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(OcrError::Unsupported(format!(
                "{name}: vision tensor must be floating point"
            )));
        }
        if tensor.shape != shape {
            return Err(OcrError::Shape(format!(
                "{name}: expected {shape:?}, got {:?}",
                tensor.shape
            )));
        }
        names.push(name);
    }
    Ok(names)
}
