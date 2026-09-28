# Packed INT4 full-page resident acceptance

Compare current native packed W4A8 with pinned llama.cpp Q8_0, CPU-only on M4 Pro.
First run a four-page native pilot with the benchmark's matched-newline prompt;
retain all outcomes and establish EOS/output quality before timed repetition.
Then use ABBA engine order: native, llama, llama, native. Each process gets one
warmup and five measured requests per page, totaling two warmups and ten measured
requests per page per engine. The same seed42 shuffled schedules, four-page
hashed corpus, 8 CPU workers, BLAS8, concurrency1 and cap8192 apply throughout.
No other inference, compilation or tests run alongside timed processes.

Native default original prompt remains unchanged; this experiment explicitly
uses `<image>\ndocument parsing.` to match llama MTMD. Native prefix+128 generated
KV and ngram35/window128 differ from llama full attention/no equivalent processor.
Original-prompt INT4 accuracy selection is already recorded separately. These
synthetic pages are not unseen quality validation. This is an end-to-end system
comparison with disclosed precision/algorithm differences, not isolated kernels.

Report all raw outputs, finish reasons, token counts, CER, median/p95 wall, TTFT,
decode rates, exact-child lifetime peak RSS and startup separately. Keep every
outlier. EOS does not establish OCR quality; disclose any quality or length
mismatch alongside speed ratios. Validate no cached llama prefix. Compare fresh
runs; earlier Q8 native results provide historical context only. Startup values
from resident processes are observations, not a repeated cold-start benchmark.

Full goal still includes broader quality, precision/fresh-process comparisons,
continued optimization and whole-branch review. This experiment cannot by itself
establish universal superiority or complete the goal.

## Pilot finding and changed next action

All four pilot requests reached EOS, but table returned zero tokens/CER1 with
packed Q4 and the matched newline. Other three CER0. The ABBA comparison is not
launched against that failed quality candidate. This is a verified quality gate,
not an unavailable hardware blocker. Original-prompt table remains correct.

A first-token diagnostic holds each Q4 hidden state fixed and varies only its
output head. For matched table, Q4 chooses EOS (18.831995) over det (18.223175);
Q8 head chooses det (18.106924) over EOS (17.983084). BF16 head also chooses det,
with a very small margin. Packed and unpacked Q4 reproduce all six diagnostic
rows exactly. This localizes a sufficient repair at this boundary; it does not
prove complete-generation quality or universal robustness.

Prepare a diagnostic Q4-in-I8 overlay with only lm_head weights/scales replaced
from the verified Q8 artifact. Independently byte-check every tensor against its
intended source. Run eight original-prompt pages and four matched-prompt pages.
Only after this gate, design packed hybrid storage/runtime support and benchmark
it. Physical I8 simulation is solely a quality experiment; no packing speed claim.

## Hybrid passes parity; resume repeated comparison

The packed hybrid format is now implemented and verified in
`../2026-09-28-hybrid-int4/`. All twelve complete outputs match its simulation
exactly, and all sixteen fixed-work logit hashes agree. `run_repeated.py` checks
those gates, verifies executable/model/projector/overlay identities, then runs
native-hybrid, llama-Q8, llama-Q8, native-hybrid. The original ABBA sampling and
matching rules above remain unchanged. All-Q4 failed pilot evidence stays intact.

## Host conditions during the live repeated run

Llama phase1 latency rose from roughly13–17s to33–41s while outputs remained
complete. Read-only host snapshots show about10GB of swap in use. Across the
recorded85.36s UTC interval, host-wide counters increased by6717 pageins,1966
pageouts,308 swapins,501613 compressions and529792 decompressions (16KiB pages).
These observations are not attributable to the benchmark child alone and do
not prove a single cause. Preserve all samples and per-phase results; do not
interpret the aggregate as isolated kernel performance. The same live runner
continues; no restart or background inference/build was introduced.

## Actual stop condition

ABBA interrupted in phase2 because the available Mac reached2% battery and then
1%, discharging. `power-interruption.json` records SIGTERM to verified owned
llama PID89694/processgroup89694/parent89095. Driver exec33128 is TERMINALexit1;
phase2 reports incomplete4/24 collected (three completed warmups and interrupted
fourth), no forced kill; phase3 neverstarted. Preserve this series as incomplete.
Heavy work requires AC power. Start any future full series in a distinct folder
and capture initial power state; do not relabel this as all96 requests complete.
