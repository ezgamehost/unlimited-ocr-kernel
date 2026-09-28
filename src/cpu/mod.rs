use crate::error::{OcrError, Result};
use std::sync::Arc;
mod accelerate;
mod neon;
pub mod ops;
pub mod quantized;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Neon,
    Accelerate,
    Auto,
}

pub struct CpuContext {
    pub backend: Backend,
    threads: usize,
    scratch_limit: usize,
    pub(crate) pool: Arc<rayon::ThreadPool>,
}

impl CpuContext {
    pub fn new(backend: Backend, threads: usize, scratch_limit: usize) -> Result<Self> {
        if threads == 0 || scratch_limit == 0 {
            return Err(OcrError::Resource(
                "threads and scratch limit must be nonzero".into(),
            ));
        }
        if backend == Backend::Neon && !cfg!(target_arch = "aarch64") {
            return Err(OcrError::Unsupported("NEON requires AArch64".into()));
        }
        if backend == Backend::Accelerate && !cfg!(target_os = "macos") {
            return Err(OcrError::Unsupported("Accelerate requires macOS".into()));
        }
        Ok(Self {
            backend,
            threads,
            scratch_limit,
            pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| OcrError::Resource(format!("CPU worker pool: {e}")))?,
            ),
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Keep a sequence of small parallel kernels on one pool worker, avoiding
    /// an external-thread handoff for every matrix. Nested calls are supported.
    pub fn run<R: Send>(&mut self, operation: impl FnOnce(&mut Self) -> R + Send) -> R {
        let pool = Arc::clone(&self.pool);
        pool.install(|| operation(self))
    }

    pub(crate) fn check_elements(&self, count: usize) -> Result<()> {
        if count.checked_mul(4).is_none_or(|n| n > self.scratch_limit) {
            return Err(OcrError::Resource("operation exceeds scratch limit".into()));
        }
        Ok(())
    }
}
