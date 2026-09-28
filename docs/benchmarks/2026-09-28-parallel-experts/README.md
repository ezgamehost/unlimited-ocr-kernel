# Parallel selected experts during decode

The previous decoder evaluated each of six selected experts in turn, splitting
every projection across the Rayon pool. This change schedules selected experts
concurrently with serial inner projections. Outputs are accumulated in the
original expert-index order. Prefill remains unchanged; low scratch limits
retain the previous scheduling. Worker contexts share the existing pool and
divide its scratch limit rather than creating additional threads.

BF16 dense-page pilot, eight workers, fresh process, 2048-token cap:

| Scheduling | Process wall (s) | Inference (s) | Prefill (s) | Decode (s) |
|---|---:|---:|---:|---:|
| Parallel experts, first run | 43.41 | 41.718 | 10.178 | 19.044 |
| Previous binary, following run | 46.97 | 44.662 | 6.692 | 25.608 |

Both produce 1649 tokens and zero normalized CER. The parallel output matches
the preceding implementation and earlier BF16 native pilot token-for-token. These are single samples, run
candidate first and baseline second, with uncontrolled filesystem cache and
memory pressure. Prefill variation makes the total-time difference weaker
evidence than the decode comparison. This is not a final repeated benchmark.

The baseline binary was copied from commit `0a81dba` before rebuilding. The first
candidate used the same schedule as the final version, with a less conservative
scratch reservation. The final reservation additionally covers expansion of
F16/F32 expert matrices and dynamic activation buffers; it does not affect this
1 GiB-scratch run. Both variants passed the original BF16 decoder reference.

Validation includes layers 0, 1 and 11 at prefill and 300 teacher-forced decode
positions across ring wraps. A separate original-weight test compares one
worker, eight workers, and eight workers with only 32 KiB scratch, requiring
identical layer outputs. The small-scratch case exercises serial fallback.

The prior INT8 pilots and their quality limitations are in
`../2026-09-28-int8-runtime`. New resident results retain the same English,
Chinese, table, dense order, independent request caches and generation settings.
Load time is separately recorded; it must be included when comparing fresh
processes. No comparison here establishes an overall win against llama.cpp.

| Page | Weight-only total / decode (s) | Dynamic activation total / decode (s) | CER, both |
|---|---:|---:|---:|
| English | 18.055 / 2.883 | 17.606 / 2.417 | 0 |
| Chinese | 18.777 / 3.607 | 17.049 / 2.770 | 0 |
| Table | 16.950 / 2.849 | 16.507 / 2.380 | 0.003984 |
| Dense | 34.787 / 19.984 | 30.651 / 15.887 | 0 |

Each mode's four outputs are token-for-token identical to its previous
serial-expert pilot. All finish at EOS. Weight-only dense decode was previously
22.508 seconds; dynamic activation dense decode was 26.741 seconds. The new
schedule makes dynamic activation quantization beneficial in this pilot despite
its previous scheduling disadvantage. These single samples still need repeated,
matched-process verification. Startup remains 17.865 seconds for weight-only and
19.973 seconds for dynamic activation; it is excluded from the table and included
in the retained process totals. Vision still takes roughly 11–12 seconds/page.
