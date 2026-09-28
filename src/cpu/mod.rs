use crate::error::{OcrError, Result};
mod accelerate;
mod neon;
pub mod ops;

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
    pub(crate) pool: rayon::ThreadPool,
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
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| OcrError::Resource(format!("CPU worker pool: {e}")))?,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub(crate) fn check_elements(&self, count: usize) -> Result<()> {
        if count.checked_mul(4).is_none_or(|n| n > self.scratch_limit) {
            return Err(OcrError::Resource("operation exceeds scratch limit".into()));
        }
        Ok(())
    }
}
