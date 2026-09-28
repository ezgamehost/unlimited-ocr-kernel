# Buffered checksum reads

Checkpoint hashing now uses a 1 MiB buffer and the same retained file handle
that backs the mapping. A mutex protects the shared file offset; each check
seeks to zero, verifies the full mapped length and rejects extra trailing data.
Read failures are returned. Atomic pathname replacement cannot redirect the
checksum to a different inode. Tensor access remains memory-mapped.

Both complete SHA256 checks remain mandatory for INT8 overlays. The source is
checked before the overlay, leaving the overlay most recently read when its
matrices are copied. Version, matrix-set, dtype, shape, range and scale checks
are unchanged. No validation is skipped or cached across processes.

Three alternating load-only runs, eight workers, dynamic INT8, same model files:

| Implementation | Load samples (s) | Median (s) |
|---|---|---:|
| Previous mapped hashing | 19.819, 17.098, 16.960 | 17.098 |
| Buffered, source first | 5.233, 5.615, 5.757 | 5.615 |

Each process loads the full model, receives EOF on JSONL stdin and exits without
inference. Order is mapped/streamed, streamed/mapped, mapped/streamed. Commands,
binary hashes and `/usr/bin/time -l` records are in `load-stream-pairs/`. The old
binary was saved from `0a81dba`; later expert/activation changes are not exercised
by this load-only comparison. Filesystem cache and machine memory pressure were
not reset. This is a startup diagnostic, not the final repeated OCR benchmark.

The first buffered experiment retained overlay-first ordering: English load was
11.213 seconds and process wall time 27.69 seconds. With source-first ordering,
load was 5.514 seconds, resident inference 15.272 seconds and process wall time
20.87 seconds. Both output exactly the preceding dynamic-INT8 English token IDs
and have zero normalized CER. Raw results are retained here. Startup is included
in the process figure; it is not included in result phase timings.

Tests cover multi-buffer content, repeated and concurrent checks, original-inode
binding after pathname replacement, corruption, source mismatch and malformed
overlays. The full Rust suite and Clippy were rerun for this change.
