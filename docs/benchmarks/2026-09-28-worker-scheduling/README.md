# Decode worker scheduling pilot

Each full decode step now stays on a Rayon worker. Its parallel BF16 GEMVs
avoid repeated handoffs from the caller thread. Matrices with at most 131072
weights run serially; larger matrices use bounded task splitting. Arithmetic
and stored weights are unchanged.

A real decode profile showed frequent scheduler yields and waits. Reproduce
the operator comparison with `cargo run --release --example bench_gemv`.
`gemv.log` measures several actual checkpoint matrices, thread counts and
whether the loop runs inside the pool. These are hot-matrix microbenchmarks,
not full-model throughput. Eight workers performed best for the tested expert
matrix, while eight and ten were similar for vocabulary projection.

The dense-page pilot used eight workers explicitly:

```sh
target/release/uocr --image benchmarks/corpus/dense.png --mode gundam --max-tokens 2048 --threads 8 --json
```

| Metric | Direct KV, 10 workers | Worker scheduling, 8 workers |
|---|---:|---:|
| Process wall seconds | 65.635 | 47.031 |
| Decode seconds | 40.588 | 24.138 |
| Output tokens | 1649 | 1649 |
| CER | 0 | 0 |
| Finish | EOS | EOS |

All 1649 token IDs match. This combined scheduling/thread-count experiment
improved decode about 1.68x in a single sequential sample. It does not isolate
each change and is not a final repeated or llama.cpp comparison. CLI defaults
remain unchanged; the eight-worker choice is explicit in the command.

The ordinary release suite, nested one/three-worker context test, BF16 scalar
and parallel tail tests, and real 300-position decoder reference checks pass.
The odd-row test now includes matrices above and below the serial threshold.
