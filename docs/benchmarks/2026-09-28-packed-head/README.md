# Vocabulary-only packing — not retained

This follow-up to the all-matrix packing experiment reorders only lm_head.weight
for dynamic INT8. Expert and attention matrices keep their original storage.
The same packed kernel and storage tests pass, and the same fixed-work decoder
benchmark runs baseline/candidate/candidate/baseline with eight workers and
VECLIB_MAXIMUM_THREADS=8 / OMP_NUM_THREADS=8. Each process has one warmup and
three measured repetitions; all sixteen output-logit checksums match exactly.

Median 512-step decode time changes from 3.735846 to 3.710154 seconds (0.7%).
Observed sample ranges overlap substantially. Prefill remains essentially
unchanged at 1.450107 versus 1.458595 seconds. This resolves the all-matrix
prefill regression but does not establish a useful end-to-end gain sufficient
to justify the extra production storage path. The candidate is reverted.

Raw timings, command/binary identities, summary, tests and exact patch are
retained. No completed-page speedup is claimed. Next, measure worker-count
sensitivity on the unchanged kernel before adding further kernel complexity.
