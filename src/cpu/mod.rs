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

    /// Share the existing pool while running an expert's inner kernels serially.
    /// Caller reserves aggregate scratch and passes a nonzero task count.
    pub(crate) fn serial_worker(&self, concurrent_tasks: usize) -> Self {
        Self {
            backend: self.backend,
            threads: 1,
            scratch_limit: self.scratch_limit / concurrent_tasks,
            pool: Arc::clone(&self.pool),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_worker_reuses_pool_but_limits_inner_parallelism_and_memory() {
        let cpu = CpuContext::new(Backend::Auto, 8, 1024).unwrap();
        let worker = cpu.serial_worker(4);
        assert_eq!(worker.threads(), 1);
        assert_eq!(worker.backend, cpu.backend);
        assert!(Arc::ptr_eq(&worker.pool, &cpu.pool));
        assert!(worker.check_elements(64).is_ok());
        assert!(worker.check_elements(65).is_err());
    }
}
