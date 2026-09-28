# Early tensor release: tested and reverted

The candidate explicitly dropped prepared image tensors after vision encoding,
projected features after insertion into prompt embeddings, and embeddings after
prefill. All arithmetic and model settings were unchanged. `identity.json` pins
both executable hashes and the candidate patch. Production source was restored
after the experiment: the candidate did not demonstrate a consistent resource
and latency improvement across the tested workloads.

The full Rust suite passed (50 passed, eight artifact/benchmark tests ignored),
clippy passed with warnings denied, and release original-reference tests passed
(eight vision and two decoder tests). Logs are in
[the verification directory](../../reviews/allocation-lifetime/).

## Four-page ABBA experiment

Each process ran one warm-up and one measured request for each corpus image.
All 32 requests reached EOS, exited normally, and produced exactly matching raw
text and token IDs by image. Per-variant medians below have only two samples per
image and do not replace final ten-repetition acceptance.

| Image | Baseline median seconds | Candidate median seconds |
| --- | ---: | ---: |
| English | 10.620335 | 10.575158 |
| Chinese | 10.943614 | 10.852023 |
| Table | 10.607323 | 10.505203 |
| Dense | 14.498752 | 14.421456 |

Process peak RSS was higher for both candidate observations:

| Run order | Peak RSS bytes |
| --- | ---: |
| Baseline A | 5,798,936,576 |
| Candidate A | 6,298,533,888 |
| Candidate B | 6,349,553,664 |
| Baseline B | 6,248,415,232 |

These are process-wide peaks, including loading and warm-ups. The variation
between baseline brackets prevents assigning every byte difference to the code.
The slight latency differences also do not establish a robust speedup.

## Maximum-crop check

Two fresh-process pairs ran in opposite orders. Every run produced exactly the
same 8,192 tokens, hit the token limit and retained CER 7.184860120680198. This is
the existing incorrect counting-text failure, not successful OCR latency.

| Order | Variant | Wall seconds | Peak RSS bytes |
| --- | --- | ---: | ---: |
| 1 | Baseline | 103.527045 | 7,109,246,976 |
| 2 | Candidate | 112.827669 | 6,977,568,768 |
| 3 | Candidate | 103.385089 | 7,008,157,696 |
| 4 | Baseline | 104.033501 | 7,093,272,576 |

Within the two pairs, candidate peak RSS was lower by 125.578 and 81.172 MiB.
The first candidate run was slower by 9.3 seconds, largely in decode; the second
was comparable to baseline. Both observations are retained. This gives evidence
of a memory benefit for this extreme input, not a general memory or latency win.
Earlier release decreases logically live tensor payload, but allocator behavior
and resident mappings determine observed process RSS.

The experiment is rejected as a default runtime optimization. Its patch and raw
records remain available for future work on allocation reuse or a broader memory
policy. No production speedup is claimed from this experiment.

Reproduce the completed four-page audit with:

```sh
.venv/bin/python docs/benchmarks/2026-09-28-allocation-lifetime/summarize.py
```

`run.sh` records the ABBA command sequence. `max32.py` and `max32.py --reverse`
record the fresh-process resource checks (use `PYTHONPATH=.` from the repository
root). Existing result directories are protected against overwrite.
