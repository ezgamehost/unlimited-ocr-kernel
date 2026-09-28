# Repeated resident measurement harness

`tools/benchmark_resident.py` runs one inference child per invocation, with a
matched image/newline prompt, explicit worker and BLAS limits, one request at a
time, and no KV/prompt reuse. Defaults are two warmup rounds and ten measured
rounds, each covering every corpus image in a seeded, shuffled order. Every
request retains raw output, token IDs, finish reason, CER and timing metadata.
The input image hashes are checked before inference.

The native JSONL reader has a deadline and byte limit and preserves partial
bytes in the process log. The llama client preserves raw SSE, accumulates token
IDs independently of chunk count, requires a terminal event and measures first
streamed output arrival. It checks zero prompt-cache reuse and 1,518 prompt
tokens. Native first-token timing comes from the internal runtime clock; llama
first-token timing includes HTTP transport. These definitions remain explicit.

The direct inference child is reaped with `wait4`, obtaining that PID's peak
RSS and CPU time rather than cumulative resources from previous child processes.
Darwin RSS is bytes; Linux RSS is converted from KiB. The server receives SIGTERM
for shutdown and is allowed to finish; forced kills and missing resources prevent
a successful protocol-completion result. Failures and truncations are retained,
with separate completion counts rather than counted as fast successful OCR.

Tests cover a real recorded SSE fixture, multiple tokens per event, incomplete
streams, warmup/sample counts, partial-line timeout retention, real child memory
allocation, forced cleanup and the full native JSONL protocol with a fixture
process. The initial full-protocol test found overly strict whitespace matching
in the startup event; parsing the JSON event fixed it. RED/GREEN logs are retained.

Smoke measurements execute one real English request in each runtime. Both reach
EOS with 250 emitted tokens and zero CER. Native wall is 11.548 seconds, llama
wall is 14.491 seconds. Measured peak RSS is 6,359,744,512 bytes for native and
7,236,763,648 bytes for llama, with normal exits and no forced kills. These are
harness-validation samples, not the final repeated comparison. The configurations
retain different precision and attention policies documented in prior reports.
