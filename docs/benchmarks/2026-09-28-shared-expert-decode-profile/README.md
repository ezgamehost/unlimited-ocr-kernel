# Accepted shared-expert runtime decode profile

The accepted CLI `1846f3dd7495e3fb0e675ce38931de1daae4bda967f6ad13bba65e885a2e93b3` completes warmed and sampled Q8 dense-page requests with the exact accepted text and 768 token IDs, both ending at EOS. Native and sampler processes exit normally, without forced kills. Power snapshots remain AC before and after.

The three-second capture still shows dynamic INT8 output computation (4,921 collapsed leaf samples), activation quantization (130), BLAS attention and substantial worker synchronization. The main thread has 1,807 samples waiting inside decoder step and 17 in token selection. Aggregate thread samples are not wall-time percentages. These measurements include profiler overhead and do not update the performance benchmark.

The remaining attribution question is the wall-time split between input projections, attention, expert execution and vocabulary projection. A separate temporary test-only stage probe records those enclosing durations and checks every logit against the preserved accepted executable. It adds no production feature.
