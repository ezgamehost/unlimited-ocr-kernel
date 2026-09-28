# Current Q8 runtime validation and comparison

The current native CLI includes the accepted GELU and head-major cache changes.
Before timing Q8 against llama.cpp Q8, twelve complete OCR cases were run with
the preserved pre-GELU/cache binary and the current binary, using the same Q8
overlay. All text, token IDs, EOS and image token counts match exactly. Both
processes exit normally. The table retains its existing one-character error
(CER 0.003984); the other ten prompt/page cases have zero character errors.

This is preservation of the Q8 baseline, not proof of unrestricted accuracy.
The separately recorded 32-crop case retains the BF16/Q8 transcript and its
0.069117 CER; the hybrid Q4 variant has a much larger regression there. See
the sibling `2026-09-28-hybrid-max32` report. The original model attribution of
the remaining BF16/Q8 stress-case errors remains pending.

The native binary SHA is
`c191d8963f450c57e9e71b17c1ad0612ad77a63ef1f3952a9f850f9747e19b57`.
The Q8 overlay SHA is
`b6afd272e6f19de970f97245c83b09e110a20acc8033f2853d0e6739e5fde441`.
`quality.py`, `baseline/` and `quality/` preserve the exact procedure and raw
proof. The runner rejects a hybrid proof when Q8 is selected before creating
an output directory or running inference (`precision-guard.json`). The updated
auditor also successfully rechecked the preceding 96-request hybrid series.

Start a new isolated AC comparison from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-packed-int4-resident/run_repeated.py \
  --native-precision int8 \
  --quality-proof docs/benchmarks/2026-09-28-current-q8/quality \
  --output-root docs/benchmarks/2026-09-28-current-q8/powered-abba-1
```

Only after all four phases finish, audit with `summarize_repeated.py` from the
same runner directory and `--input-root` pointing to that fresh series. There
is no new full-page performance claim until the complete raw audit passes.
