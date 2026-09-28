use crate::error::{OcrError, Result};

/// Token-major keys and values: immutable prefix followed by a bounded ring.
pub struct LayerCache {
    width: usize,
    window: usize,
    prefix: Option<usize>,
    generated: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerCache {
    pub fn new(heads: usize, head_dim: usize, window: usize) -> Result<Self> {
        let width = heads
            .checked_mul(head_dim)
            .filter(|&n| n > 0)
            .ok_or_else(|| OcrError::Shape("cache head dimensions".into()))?;
        if window == 0 {
            return Err(OcrError::Shape("empty cache window".into()));
        }
        Ok(Self {
            width,
            window,
            prefix: None,
            generated: 0,
            keys: Vec::new(),
            values: Vec::new(),
        })
    }

    pub fn prefill(&mut self, keys: &[f32], values: &[f32], tokens: usize) -> Result<()> {
        if self.prefix.is_some()
            || tokens == 0
            || tokens.checked_mul(self.width) != Some(keys.len())
            || keys.len() != values.len()
        {
            return Err(OcrError::Shape("invalid cache prefill".into()));
        }
        let capacity = tokens
            .checked_add(self.window)
            .and_then(|n| n.checked_mul(self.width))
            .ok_or_else(|| OcrError::Resource("cache capacity overflow".into()))?;
        self.keys
            .try_reserve_exact(capacity)
            .map_err(|e| OcrError::Resource(e.to_string()))?;
        self.values
            .try_reserve_exact(capacity)
            .map_err(|e| OcrError::Resource(e.to_string()))?;
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.prefix = Some(tokens);
        Ok(())
    }

    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        let prefix = self
            .prefix
            .ok_or_else(|| OcrError::Shape("cache must be prefilled".into()))?;
        if key.len() != self.width || value.len() != self.width {
            return Err(OcrError::Shape("decode cache vector width".into()));
        }
        if self.generated < self.window {
            self.keys.extend_from_slice(key);
            self.values.extend_from_slice(value);
        } else {
            let offset = (prefix + self.generated % self.window) * self.width;
            self.keys[offset..offset + self.width].copy_from_slice(key);
            self.values[offset..offset + self.width].copy_from_slice(value);
        }
        self.generated += 1;
        Ok(())
    }
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn tokens(&self) -> usize {
        self.keys.len() / self.width
    }
    pub fn position(&self) -> usize {
        self.prefix.unwrap_or(0) + self.generated
    }
}
