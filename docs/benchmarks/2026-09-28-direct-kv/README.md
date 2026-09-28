# Direct KV decode pilot

Single-token attention now reads the existing token-major prefix/ring cache
through checked strided matrix-vector operations. It no longer copies both
full KV tensors into head-major storage or transposes V on each layer/step.
Prefill behavior and the preserved-prefix/128-token ring policy are unchanged.

`cargo run --release --example bench_decode_attention` compares the old
copy/rearrange path with the direct path in one process. For a 1641-token cache,
the three old-path samples were 1.85–1.99 ms/layer and direct-path samples were
0.124–0.131 ms/layer. See `attention.log`. This isolates attention, not the
full decoder or OCR pipeline. Tests compare arbitrary strides against an FP64
oracle and test malformed shapes/resource limits. The existing real upstream
300-position decoder trace passes across ring wrap boundaries.

The dense-document end-to-end pilot used:

```sh
target/release/uocr --image benchmarks/corpus/dense.png --mode gundam --max-tokens 2048 --json
```

| Metric | Previous native pilot | Direct KV pilot |
|---|---:|---:|
| Process wall seconds | 106.717 | 65.635 |
| Vision seconds | 13.998 | 13.144 |
| Prefill seconds | 4.594 | 9.906 |
| Decode seconds | 86.269 | 40.588 |
| Output tokens | 1649 | 1649 |
| Character error rate | 0 | 0 |
| Finish | EOS | EOS |

Every output token ID matches the previous run. `dense.json` retains the raw
output and timings. Decode improved about 2.13x in this exploratory sample.
The runs were sequential single samples, not randomized repeated trials;
file residency and system activity were not controlled. Prefill varied despite
being unchanged. No final speedup against llama.cpp is claimed.

The crop-policy discrepancy is now traced to metadata: the saved llama log
reports `preproc_tiles: 2 - 9`, while the upstream model allows 32. At the pinned
llama commit, [the CLIP loader](https://github.com/ggml-org/llama.cpp/blob/4da6337767f973e2b4d0797e5b323d77d8565e4a/tools/mtmd/clip.cpp#L1893)
reads `clip.vision.preproc_max_tiles` when present, otherwise defaults to 9 for
this projector. A provenance-recorded projector derivative with corrected
metadata can align the cap without changing any tensor weights. That matched
baseline remains to be produced and measured.
