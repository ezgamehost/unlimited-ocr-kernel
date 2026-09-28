# llama.cpp pilot with upstream crop limit

Later audit: the pinned MTMD code adds a text newline after image embeddings.
These llama runs therefore have a 1,518-token prefix, while earlier native
pilots used 1,517. The [matched resident pilot](../2026-09-28-thread-controls/README.md)
corrects the comparison input and records the consequences. Raw historical
outputs and timings here remain unchanged. These commands also predate explicit
BLAS environment controls; the tool at `48ac31b` preserves that older behavior,
whereas the current helper records an explicit BLAS limit.

The original community projector omitted the Unlimited-OCR crop limit, causing the pinned llama.cpp loader to default to 9 tiles. This derivative adds `clip.vision.preproc_max_tiles = 32`. Every one of its 476 tensor payloads was checked byte-for-byte by SHA256; all other metadata is unchanged. The source is untouched. `projector-manifest.json` records both file hashes and tensor hashes.

Preparation (requires the pinned Python reference environment):

```sh
python -m tools.projector_metadata /tmp/unlimited-ocr-baseline/models/mmproj.gguf artifacts/mmproj-unlimited-ocr-F16-tiles32.gguf --max-tiles 32 --manifest artifacts/mmproj-tiles32-manifest.json
python -m tools.benchmark --manifest benchmarks/corpus/manifest.json --llama /tmp/unlimited-ocr-baseline/bin/llama-b11223/llama-mtmd-cli --model /tmp/unlimited-ocr-baseline/models/model.gguf --mmproj artifacts/mmproj-unlimited-ocr-F16-tiles32.gguf --output artifacts/llama-tiles32-pilot --threads 8 --warmups 0 --repetitions 1 --max-tokens 2048
```

This is a single-run exploratory pilot, not the final 2-warmup/10-repetition benchmark. CPU only, eight threads, fresh processes; model and executable provenance remain those in the original baseline report. CLI details and raw stdout/stderr are retained beside each record.

Vision-phase timing correction: the original parser counted only the first
`mtmd batch encoding done` line. Summing all three batches gives English
11.336 s, Chinese 12.230 s, table 11.868 s and dense 11.706 s. The per-page JSON
records are corrected; `vision-timing-correction.json` retains previous values
and batch durations. Raw logs, process wall time, decode time, outputs and CER
are unchanged. Earlier conversational comparisons using the first-batch vision
time should not be used as full vision comparisons.

| Page | Process wall (s) | Decode (s) | CER | Finish |
|---|---:|---:|---:|---|
| english | 18.738 | 1.230 | 0.000000 | eos |
| chinese | 18.474 | 1.599 | 0.000000 | eos |
| table | 17.821 | 1.216 | 0.003984 | eos |
| dense | 20.113 | 3.791 | 0.000000 | eos |

The logs confirm `preproc_tiles: 2 - 32`, four local rows of 310 tokens and 273 global tokens, totaling 1513 image tokens. This aligns the crop count with native upstream mode on these inputs. It does not establish pixel-identical preprocessing or precision equivalence: llama still uses Q8 decoder/F16 vision weights and full attention instead of the native preserved-prefix/128-generated-token ring. Both runtimes finish these documents; the native pilot has zero CER on the table where llama has a single O/0 error.

The native eight-worker dense pilot took 47.031 seconds versus 20.113 seconds here. Native remains slower. Later comparisons must repeat matched process modes, report quality differences, and include preparation/loading costs consistently.

The dense outputs also differ in layout granularity: llama uses 768 decode
steps and one large text region, while native emits 1649 tokens with more
layout annotations. Both transcribe all expected characters. Thus the wall-time
comparison includes different generation lengths; it cannot by itself measure
relative decoder-kernel speed or equivalent layout quality. Final reporting
must retain this distinction and include a fixed-work decoder measurement.
