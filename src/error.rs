#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid checkpoint: {0}")]
    Format(String),
    #[error("invalid tensor shape: {0}")]
    Shape(String),
    #[error("unsupported configuration: {0}")]
    Unsupported(String),
    #[error("resource limit: {0}")]
    Resource(String),
    #[error("nonfinite numerical input: {0}")]
    Numerical(String),
}

pub type Result<T> = std::result::Result<T, OcrError>;
