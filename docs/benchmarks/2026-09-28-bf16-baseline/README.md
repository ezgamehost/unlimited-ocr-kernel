# BF16 llama.cpp CPU resident baseline

All 48 requests completed at EOS, including two warm-up rounds and ten measured
rounds across the four versioned corpus images. The process exited normally with
code 0, no forced kill, no cached prompt tokens and 1,518 prompt tokens per
request. No failed, truncated or slow samples were excluded.

| Image | Median seconds | Nearest-rank p95 seconds | Median decode tokens/s |
| --- | ---: | ---: | ---: |
| English | 46.329118 | 50.166542 | 21.130019 |
| Chinese | 47.680777 | 60.011867 | 22.240789 |
| Table | 46.393534 | 49.841953 | 21.273298 |
| Dense | 68.587889 | 77.681061 | 22.410884 |

With ten measurements, nearest-rank p95 is the largest observation. Startup was
4.142697 seconds (one observation). Exact-child wait4 peak RSS was 9,835,954,176
bytes (about 9.16 GiB); total process wall time was 2,558.797028 seconds, including
warm-ups. `llama-resident/process.json` retains CPU time and fault counters too.

The decoder stores 119 BF16 tensors and 36 F32 tensors; the shared projector
stores 146 F16 tensors and 330 F32 tensors. These are weight-storage descriptions,
not claims that all computation uses that dtype. See `artifact.json` for the
pinned download revision and verified checksum. CPU-only llama.cpp b11223 uses
eight workers and eight BLAS threads, with all offloading and prompt caches
disabled. The full command, binary/runner hashes, thread environment, manifest,
shuffle schedule, raw SSE streams and decoded outputs are retained.

All twelve outputs per image are identical within this BF16 run. Normalized
transcripts match the earlier Q8 run: CER is zero for English, Chinese and dense,
and 0.00398406374501992 for the table (one character). Raw layout output differs:
English has one coordinate-token difference, Chinese has three token differences
including a heading/text label, and dense has two coordinate-token differences.
The table raw output matches Q8 exactly. Equal text CER is not full layout parity.

**Host paging was active.** See [host-context.md](host-context.md) and the retained
host snapshots. These results measure this shared-machine run; the gap from the
earlier Q8 run cannot be attributed solely to precision or kernel performance.
Native BF16 and contemporaneous precision comparisons remain pending. This
baseline does not replace the completed Q8-versus-native comparison, and it does
not establish completion of the broader optimization objective.

Reproduce the output audit from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-bf16-baseline/check_outputs.py
```

`output-validation.json` now audits all 48 requests and marks the benchmark
complete. Request latency aggregates are in `llama-resident/summary.json`.
Decode throughput uses each request's decoder step count and decode duration;
it excludes the first token produced by prefill. Streaming arrival TTFT is
available in each record and differs from the native runtime's internal clock.
