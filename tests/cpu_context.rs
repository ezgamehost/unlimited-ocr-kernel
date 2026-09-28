use uocr::cpu::{Backend, CpuContext};
#[test]
fn worker_context_is_reentrant_and_uses_the_configured_pool() {
    for threads in [1, 3] {
        let mut cpu = CpuContext::new(Backend::Auto, threads, 1024).unwrap();
        let returned = cpu.run(|cpu| {
            assert_eq!(rayon::current_num_threads(), threads);
            assert!(rayon::current_thread_index().is_some());
            cpu.run(|cpu| cpu.threads() + 7)
        });
        assert_eq!(returned, threads + 7);
    }
}
