use crate::error::{OcrError, Result};
use std::sync::Arc;
mod accelerate;
#[cfg(target_os = "macos")]
mod blas_threads;
#[cfg(test)]
mod cache_probe;
pub mod int4;
mod neon;
#[cfg(target_arch = "aarch64")]
mod neon_gelu;
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

    /// Give a wider shared expert spare capacity in the existing pool.
    /// Other expert tasks retain one worker each; aggregate scratch is unchanged.
    pub(crate) fn shared_expert_worker(
        &self,
        concurrent_tasks: usize,
        shared_experts: usize,
    ) -> Self {
        let mut worker = self.serial_worker(concurrent_tasks);
        worker.threads = self
            .threads
            .saturating_sub(concurrent_tasks.saturating_sub(1))
            .clamp(1, shared_experts.max(1));
        worker
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
    fn shared_expert_uses_spare_workers_with_same_memory_partition() {
        for (threads, expected) in [(1, 1), (4, 1), (8, 2), (10, 2)] {
            let cpu = CpuContext::new(Backend::Auto, threads, 7 * 1024).unwrap();
            let worker = cpu.shared_expert_worker(7, 2);
            assert_eq!(worker.threads(), expected);
            assert_eq!(worker.backend, cpu.backend);
            assert!(Arc::ptr_eq(&worker.pool, &cpu.pool));
            assert!(worker.check_elements(256).is_ok());
            assert!(worker.check_elements(257).is_err());
        }
    }

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
