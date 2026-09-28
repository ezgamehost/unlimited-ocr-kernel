# Native document pilot

Single fresh-process runs on M4 Pro. Original BF16 weights, FP32 activations, FP64 LayerNorm accumulation, upstream gundam crop policy. Each run used:

```sh
target/release/uocr --image benchmarks/corpus/english.png --mode gundam --max-tokens 2048 --json
```

Substitute the other three corpus image paths. Raw OCR text, token IDs, timing and finish reason are retained in each JSON record. CER uses the same layout/whitespace normalization as the llama pilot. This synthetic corpus is not a production accuracy evaluation.

| Page | Process wall (s) | Vision (s) | Prefill (s) | Decode (s) | Output tokens | CER | Finish |
|---|---:|---:|---:|---:|---:|---:|---|
| english | 43.873 | 14.154 | 11.837 | 17.047 | 250 | 0.0000 | eos |
| chinese | 31.776 | 13.409 | 2.534 | 14.268 | 289 | 0.0000 | eos |
| table | 29.741 | 12.656 | 2.363 | 13.428 | 247 | 0.0000 | eos |
| dense | 106.717 | 13.998 | 4.594 | 86.269 | 1649 | 0.0000 | eos |

All four complete without character errors. Performance remains far below the requested goal, particularly sustained decode. No performance victory is established.

The saved llama pilot uses Q8 decoder/F16 vision weights and 903 image tokens for these pages; this upstream native policy uses 1513. Precision, crop policy, attention semantics, loading and file residency must be aligned or explicitly separated before making comparative speed claims. No other task compilation or reference inference ran during this pilot; unrelated system activity was not controlled. Final evaluation still requires warmups, repeated measurements, matched representations and broader inputs.

Implementation at `311ce18`; this report contains exploratory evidence, not the final benchmark.
