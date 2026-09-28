use crate::error::{OcrError, Result};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn element_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| OcrError::Shape("element count overflow".into()))
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if element_count(&shape)? != data.len() {
            return Err(OcrError::Shape("shape does not match storage".into()));
        }
        Ok(Self { shape, data })
    }

    pub fn validate(&self) -> Result<()> {
        if element_count(&self.shape)? != self.data.len() {
            return Err(OcrError::Shape("shape does not match storage".into()));
        }
        Ok(())
    }
}
