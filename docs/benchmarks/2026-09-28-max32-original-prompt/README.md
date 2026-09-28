# Maximum-crop input with the original native prompt

The unchanged native runtime was run on the same hashed 640×20,480 image with
its product default, `<image>document parsing.`, rather than the newline form
used in the earlier controlled llama comparison. Both precisions validate 3,793
image tokens, exit normally and reach EOS after 930 emitted tokens.

| Precision | Fresh-process wall seconds | Peak RSS bytes | CER |
| --- | ---: | ---: | ---: |
| Dynamic INT8 | 35.723759 | 6,856,802,304 | 0.06911684037301152 |
| BF16 weights / FP32 compute | 42.126395 | 7,744,634,880 | 0.06911684037301152 |

The normalized transcripts match across precisions, though raw layout text
contains differences. Both omit sections 01 and 02, recognize sections 03–32,
and repeat section 26 at the end. This is partial OCR with remaining errors,
not full text accuracy. These are one-shot resource/diagnostic observations,
not repeated performance estimates or a universal process-memory bound.

The original prompt avoids the earlier 8,192-token counting loop. The
[receipt diagnostic](../2026-09-28-receipt-diagnostic/README.md) independently
shows first-token prompt sensitivity in the original upstream CPU model, but
that receipt result is not a reference proof for this much larger input.
The remaining section omissions/repetition still need an original-model
comparison before attributing their cause. Dynamic quantization alone does not
explain them because BF16 reproduces the normalized transcript.

All prior matched-newline results and failures remain retained. No image, model,
runtime or crop policy was altered here. Source/binary and input hashes,
commands, full outputs and exact-child resource records are under the two run
directories. Reproduce from the repository root using
`docs/benchmarks/2026-09-28-receipt-diagnostic/native_default.py` with `PYTHONPATH=.`,
`--image docs/benchmarks/2026-09-28-max32/portrait32.png`,
`--truth docs/benchmarks/2026-09-28-max32/portrait32.txt`, and a new `--output`
directory. Omit `--precision` to run both configurations.
