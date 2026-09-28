# Parallel vision activations

SAM GELU and CLIP QuickGELU now split independent elements across the existing
CPU pool. Inputs below 16384 elements, one-worker execution, and the scalar
backend retain serial evaluation. Each worker uses the existing formula;
there is no approximation change, new allocation, or additional worker pool.

On the 16,777,216-element operator workload (including input copying), eight
workers reduced GELU from about 0.133 seconds to 0.019–0.021 seconds and
QuickGELU from about 0.019 seconds to 0.0045–0.0051 seconds. These are operator
measurements, not whole-model speedups. `bench_activations` reproduces them.

The numerical test requires identical finite result bits over empty/tiny inputs,
the dispatch threshold, uneven worker tails, signed zero, extreme inputs and
infinities; NaNs must remain NaNs. All three original-weight vision checks pass
unchanged, including global SAM/CLIP/projected stages and full 12-crop merged
embeddings. Their log is retained here.

Two arithmetic-changing experiments were rejected. Vector exponentials plus
the [A&S 7.1.26 erf approximation](https://personal.math.ubc.ca/~cbm/aands/page_299.htm)
passed operator error checks and ran quickly, but failed the existing full
vision tolerance. A generated degree-24 FP64 polynomial reduced standalone
approximation error further but also failed the full reference. Restoring
original QuickGELU while keeping that polynomial still failed. Neither
approximation ships in the runtime; sources/patches and failing reference logs
are retained to make those experiments reviewable. No tolerance was relaxed.

The page pilot uses eight workers, dynamic activation INT8, a resident model,
the same English/Chinese/table/dense request order, and a 2048-token cap as
`../2026-09-28-parallel-experts`. Load cost remains separately reported. These
are single samples with uncontrolled file-cache and memory-pressure state;
they do not replace the final repeated, matched-process benchmark.

| Page | Previous vision (s) | New vision (s) | Previous total (s) | New total (s) |
|---|---:|---:|---:|---:|
| English | 11.645 | 9.313 | 17.606 | 15.226 |
| Chinese | 11.313 | 8.997 | 17.049 | 14.712 |
| Table | 11.161 | 9.108 | 16.507 | 14.421 |
| Dense | 11.681 | 9.436 | 30.651 | 28.504 |

All four outputs have identical token IDs to the preceding dynamic-INT8 pilot
and finish at EOS. CER remains zero except for the previously reported table
error (0.003984). The loader took 26.557 seconds in this run; the totals above
are resident inference and exclude it. Full phase timings and raw outputs are
retained beside this report. The small kernel patch is kept because it improves
measured page latency without changing arithmetic or observed output.
