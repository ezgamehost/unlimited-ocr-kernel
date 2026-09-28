use crate::error::{OcrError, Result};

/// Immutable prefix and bounded ring, initially token-major; optionally packed
/// into contiguous per-head storage after prefill.
pub struct LayerCache {
    width: usize,
    head_dim: usize,
    head_major: bool,
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
            head_dim,
            head_major: false,
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
        if self.head_major {
            let capacity = prefix + self.window;
            let slot = prefix + self.generated % self.window;
            for head in 0..self.width / self.head_dim {
                let offset = (head * capacity + slot) * self.head_dim;
                let input = head * self.head_dim;
                self.keys[offset..offset + self.head_dim]
                    .copy_from_slice(&key[input..input + self.head_dim]);
                self.values[offset..offset + self.head_dim]
                    .copy_from_slice(&value[input..input + self.head_dim]);
            }
        } else if self.generated < self.window {
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

    /// Convert once to [head, prefix+window capacity, head_dim] storage.
    /// Active tokens retain their original physical order. Allocation failure
    /// leaves both original buffers intact; no permanent duplicate is retained.
    pub fn pack_heads(&mut self) -> Result<()> {
        let prefix = self
            .prefix
            .ok_or_else(|| OcrError::Shape("cache must be prefilled".into()))?;
        if self.head_major {
            return Ok(());
        }
        let capacity = prefix + self.window; // checked by prefill
        let elements = capacity * self.width;
        let pack = |input: &[f32]| -> Result<Vec<f32>> {
            let mut output = Vec::new();
            output
                .try_reserve_exact(elements)
                .map_err(|e| OcrError::Resource(e.to_string()))?;
            output.resize(elements, 0.);
            for head in 0..self.width / self.head_dim {
                for token in 0..self.tokens() {
                    let dest = (head * capacity + token) * self.head_dim;
                    let source = token * self.width + head * self.head_dim;
                    output[dest..dest + self.head_dim]
                        .copy_from_slice(&input[source..source + self.head_dim]);
                }
            }
            Ok(output)
        };
        let keys = pack(&self.keys)?;
        let values = pack(&self.values)?;
        self.keys = keys;
        self.values = values;
        self.head_major = true;
        Ok(())
    }

    pub fn is_head_major(&self) -> bool {
        self.head_major
    }

    /// Physical storage; packed caches include unused capacity after each head.
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn tokens(&self) -> usize {
        if self.head_major {
            self.prefix.unwrap_or(0) + self.generated.min(self.window)
        } else {
            self.keys.len() / self.width
        }
    }
    pub fn position(&self) -> usize {
        self.prefix.unwrap_or(0) + self.generated
    }
}
