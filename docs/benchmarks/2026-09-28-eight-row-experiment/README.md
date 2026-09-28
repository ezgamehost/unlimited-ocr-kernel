# Eight-row SDOT experiment — rejected

An eight-row, row-major kernel shares activation loads across twice as many
outputs as the retained four-row kernel. Its independent scalar-oracle test
passes bitwise across model projection widths, unaligned storage, integer
extremes and varied scales. Full Rust tests pass. RED/GREEN logs and the exact
experimental patch are retained; the candidate was confined to unit-test code.

Alternating single-thread microbenchmarks use expert up/down and vocabulary
matrix shapes. Checksums match. The expert shapes are essentially unchanged;
the vocabulary head is slower with eight rows. See summary.json and raw timings.
This provides no reason to add a production path or run expensive full-model
acceptance measurements. The experiment is reverted; production dispatch stays
unchanged. Timings are short operator pilots, not end-to-end performance claims.

Next experiment: four-row block-interleaved weights and scales to improve
sequential access, initially as a standalone candidate. Any packed production
storage must replace the row-major copy or use a bounded cache, not double the
whole quantized model in memory.
