# Variable-token resident adapter check

Both real runtimes processed a mixed manifest in one resident process: the
440×133 logo (273 image tokens / 278 prompt tokens) and original English page
(1,513 / 1,518). Each process exited 0 without a forced kill, and every request
reached EOS. llama reported zero cached prompt tokens. `prefix_tokens` is null
for this mixed manifest; `expected_tokens_by_image` retains both expectations.
The runner's default count remains unchanged for existing manifests.

This is a two-request adapter smoke check, without warm-ups or repeated timing;
it is neither held-out validation nor a performance acceptance benchmark. The
Python suite passed 30 tests with two skips. Regressions reproduce the previously
rejected valid single-view response and retain rejection of mismatched image
counts, wrong server prompt lengths and cache reuse.

English CER is zero. Both runtimes emit the same logo output:

```
title [0, 0, 999, 999]<|/det|>Bai du 百度
```

The unmatched detection closing tag is retained by the scorer, producing CER
3.875 against `Baidu 百度`. This is not a quality success; EOS only establishes
completion. The adapter change does not alter text normalization or hide the
malformed layout output. Raw streams, decoded text, request records, hashes and
exact process commands are retained under `native/` and `llama/`.
