# Larger SAM attention tiles

SAM's biased attention now permits 1024-query tiles, while unbiased CLIP and
decoder attention retain 128-query tiles. Both use the same implementation and
shrink tiles to fit the existing scratch limit, including output and transposed
value storage. This reduces matrix calls and row-task dispatches in SAM without
changing its bias, scaling, masking or softmax formulas.

The accepted candidate reduced median full-Gundam vision latency from **7.1763s
to 6.1927s (13.7%)** in `sam-abba/`. Four sequential processes ran baseline,
candidate, candidate, baseline; each had one warmup and three measured encodes
of the English image, including all crops. Each variant has six measured values.
Eight Rust workers and eight BLAS threads were used on the shared M4 Pro with
AC checked before each process. No concurrent build or profiling ran during
this comparison. The observed ranges were 7.142–7.198s versus 6.176–6.261s.

All 16 outputs had shape [1513, 1280] and the same FP32 embedding SHA256:
`a2b4ecf2092508904b5bd359c86fbce682c801ffd7cd93336658b25c7acc7213`.
Hashing and finite checks occur outside the measured encode interval. These
are vision timings, not a new end-to-end llama.cpp comparison or a universal
hardware claim. Full-page timings in the quality run are single observations.

## Why only biased attention changed

An initial forward/reverse experiment enlarged the shared tile cap everywhere:

| Query cap | Median full-vision seconds | English embedding hash |
|---|---:|---|
| 128 | 7.1950 | baseline |
| 256 | 6.6906 | baseline |
| 512 | 6.3830 | different |
| 1024 | 6.2480 | different |

The all-1024 candidate passed the existing numerical reference tolerances but
failed the exact complete-OCR gate on 4 of 12 cases (`quality-1024/`). All cases
reached EOS with unchanged generation lengths, recognized text and CER 0, but
eight bounding-box coordinates moved by one unit. The raw failure and coordinate
audit are retained. No tolerance was relaxed to turn that gate green.

Restoring unbiased attention's 128-query tiles while keeping SAM's 1024-query
tiles restored exact English embeddings, followed by exact text/token IDs on
all 12 OCR cases (`quality-sam1024/comparison.json`). This isolates the observed
drift to the changed unbiased tiling in this experiment; it does not promise
bitwise equality for every possible image or BLAS implementation.

The first SAM-only pilot was used to inspect hashes. Its timing is exploratory:
an unrelated coordinate audit was launched around the completion poll. Only the
subsequent isolated `sam-abba/` measurements support the accepted speed claim.

## Validation

- Standard Rust suite: 58 passed, 9 intentionally ignored. Existing original
  decoder and vision checks were then run explicitly: 2 decoder and 8 vision
  checks passed, including global/local features and merged crop embeddings.
- A new biased-attention test checks absolute query offsets across 1031 queries
  with both 8704-byte and 1 MiB scratch budgets, scalar and accelerated backends.
  One-hot bias selects literal expected values. A deliberate zero-offset mutation
  failed with 3 versus 12; restoring the actual offset passed the full suite.
- The first Clippy pass flagged a constant-sized chunk iterator in that test.
  It was changed to the array-chunk iterator; final Clippy, format, and ops
  checks are recorded separately. This was a test-only lint correction.
- All 12 complete OCR outputs are exact to the previously accepted hybrid
  outputs, with EOS and image-token counts preserved. The cases cover eight
  synthetic pages plus four matched-prompt primary cases, not unseen production
  documents. Exact parity carries forward their existing CER 0 observations.

## Reproduction and artifacts

`build.py` records the initial experiment: start with `src/cpu/ops.rs` from
commit `c2768bb` (the 128-cap implementation), and the benchmark source snapshot
`benchmark-source.rs.txt` as `examples/bench_vision_tiles.rs`. It compiles the four
variants and restores that source after each build. The accepted implementation
has since split the biased/unbiased caps, so the historical builder deliberately
rejects it rather than silently running a different experiment. Use an isolated
checkout to rebuild the historical variants. Source and executable hashes are
in `builds.json`; the snapshot preserves the exact pre-format benchmark source.

`run.py` measures the four original variants; `run.py --sam-only` measures the
baseline and the executable identified by `sam1024-identity.json`. Both require
fresh output directories and verify binary/model hashes. Commands run with
`PYTHONPATH=. .venv/bin/python` from the repository root. Existing result
directories are protected from overwriting.

The accepted production CLI identity and full request/response evidence are in
`quality-sam1024/process.json`. The previous complete llama comparison remains
in `../2026-09-28-packed-int4-resident/powered-abba-1/`; do not apply the vision
percentage directly to those full-page speed ratios.
