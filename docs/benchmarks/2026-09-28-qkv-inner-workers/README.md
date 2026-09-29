# Rejected additional QKV inner workers

Three concurrent single-token QKV projections were assigned floor(workers/3)
inner workers instead of one. At eight total workers this gives each projection
two workers in the same Rayon pool, with unchanged aggregate scratch accounting.
The policy unit test fails before the helper exists, then passes across worker
counts 1/3/4/6/8/10/14 and memory boundaries.

The AC-powered baseline/candidate/candidate/baseline series records one warmup
and three measurements per process, six measured samples per variant. All sixteen
complete-logit hashes agree and all four processes exit normally without forced
kills. The native fixed-work input is 1517 prefix tokens plus 512 teacher-forced
steps, with eight Rust/BLAS/OMP workers. Decode includes full-logit hashing.

Median decode is **3.273513 → 3.314873 seconds (1.26% higher)**. Measured ranges
overlap (3.261751–3.306502 versus 3.284390–3.337373). Prefill is
1.388466 → 1.380906 seconds, also with overlapping ranges. This supplies no
performance benefit supporting the scheduling change. Additional scheduling
cost is a plausible explanation, not isolated by this experiment.

The candidate is rejected and production source restored. BF16 and weight-only
performance were not measured for this rejected proposal. No whole-page or
accuracy improvement is claimed. `experiment.patch`, RED/GREEN logs, preserved
binary identities, source hashes, power checks and raw samples retain the work.
The baseline is accepted exact-quantizer source `5ce97a5`; the CLI is untouched.
