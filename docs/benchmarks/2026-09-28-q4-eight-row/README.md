# Eight-row INT4 kernel — rejected

The candidate shares each activation block across eight packed INT4 rows,
instead of four. Two FP32 accumulators preserve the existing per-block scale
multiplication and addition order. The Q8 head, weight format, precision and
decoder policies are unchanged. A four-row remainder uses the original kernel;
other tails and unsupported targets retain scalar evaluation.

The new direct test first failed because the eight-row function was missing,
then passed bitwise against two calls to the accepted four-row kernel. It covers
32, 64, 96, 896, 1280, 1792 and 6848 columns with signed codes, signed zeros and
varied scales. Expanded integration cases cover eight-row groups and tails,
partial column blocks and parallel dispatch. The full candidate suite passed
61 tests with nine ignored. Raw logs and the full candidate patch are retained.

The isolated fixed-work ABBA comparison uses the hybrid overlay, eight workers,
eight-thread BLAS environment limits, a 1,517-token prefix and 512 teacher-forced
decode steps. Each process has one warmup plus three measurements. All sixteen
repetitions retain logit SHA-256
`c129d845d9d0f6b2b1f7bc1aaa7741d1a9fb95b428a1016f519df94acb862f3a`.
All four processes exit normally. AC power is checked before each phase; no
builds or profiling overlap timing. The host is not otherwise isolated.

| Metric | Four rows | Eight rows |
|---|---:|---:|
| Median decode including checksum | 3.404646s | 3.438481s |
| Observed decode range | 3.369944–3.415918s | 3.413910–3.451258s |
| Median prefill | 1.614950s | 1.625147s |

The candidate is about 0.99% slower at median decode and is rejected. Sharing
activation loads does not establish a benefit on this workload. This experiment
does not identify the hardware cause of the regression. No full-page speedup
or broader quality claim follows from these measurements. Production changes
and candidate-only tests are reverted; the established GELU CLI is restored.

`abba/` contains executable/source identities, the complete patch and every
raw observation. To reproduce, start from `31821dd`, apply `candidate.diff`,
build/run the Rust suite to `tests.log`, then run `run.py` with fresh output and
artifact paths. The baseline executable identity comes from the preceding
Q8-scheduling experiment, whose baseline includes the accepted GELU change.
