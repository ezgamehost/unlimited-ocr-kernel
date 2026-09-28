# Rust worker-count sweep

Fixed-work native decoder benchmark: original row-major dynamic INT8 kernel,
1,517-token prefix, 512 teacher-forced steps and full-logit checksum. Each
process runs one warmup and three measured repetitions. BLAS and OMP caps stay
at eight; only the Rust worker count changes. This isolates decoder scheduling,
not an overall single-thread CPU configuration or a completed OCR benchmark.

| Rust workers | Measured samples | Median decode (s) | Median prefill (s) |
|---:|---:|---:|---:|
| 1 | 3 | 7.007036 | 1.652633 |
| 4 | 3 | 3.875408 | 1.439713 |
| 8 | 6 | 3.763752 | 1.456821 |
| 10 | 3 | 4.128471 | 1.476684 |
| 14 | 3 | 4.650717 | 1.524357 |

Order: 8, 4, 10, 1, 14, 8. The two eight-worker process medians are 3.756189
and 3.773341 seconds (less than 0.5% difference). All 24 repetitions have the
same logit checksum, 188943474296580907; all six processes exit 0. The full
warmup results, timings and binary identities are retained.

Eight workers is the best tested decoder configuration here; more workers do
not improve it. This supports retaining the existing eight-worker benchmark
setting. It does not establish the best worker/BLAS combination for vision or
for held-out complete pages. No runtime default is changed based on this sweep.

Reproduce one configuration after building release unit tests:

```sh
UOCR_BENCH_THREADS=4 VECLIB_MAXIMUM_THREADS=8 OMP_NUM_THREADS=8 \
  target/release/deps/uocr-17f9463a030211d5 \
  fixed_work_decoder_benchmark --ignored --nocapture --test-threads=1
```

The unit executable filename may change; use the path printed by
`cargo test --release --lib --no-run`. `UOCR_BENCH_THREADS` only controls the
ignored benchmark harness; its default remains eight.
