# Repeated frozen-page comparison

Both configurations completed all 48 requests and exited 0 without forced
termination: two warm-ups plus ten measured requests per image, eight workers
and eight BLAS threads, no prompt-cache reuse, and an 8,192-token cap. All twelve
outputs per image are identical within each configuration. Every crop/prompt
count, record index, input hash, schedule and CER was audited by `summarize.py`.

| Page | Native median / p95 seconds | llama median / p95 seconds | Native / llama CER |
| --- | ---: | ---: | ---: |
| Wide report | 5.491466 / 5.746843 | 5.442274 / 6.084925 | 0 / 0 |
| Bilingual notice | 6.920652 / 7.110349 | 7.171748 / 7.348489 | 0 / 0 |
| Archive register | 17.828841 / 18.042190 | 20.598118 / 20.905751 | 0 / 0.438795337 |
| Receipt | 2.166416 / 2.423372 | 2.383797 / 2.641159 | 1 / 1 (empty) |

P95 is nearest-rank and therefore the maximum of ten samples. On the two pages
with nonempty zero-CER output in both configurations, the llama/native median
ratios are 0.991042 (wide report) and 1.036282 (bilingual notice). This is a small
mixed result: no broad speed advantage is established on those two page shapes.
Archive output lengths differ (1,790 vs 2,497 tokens), with repetitions/reordering
in llama. Receipt outputs are empty. No quality-matched speed ratio is reported
for either failing comparison; their raw timings remain visible above.

Peak exact-child RSS was 6,251,036,672 bytes native versus 9,090,121,728 bytes
llama. Startup was 7.112148 versus 4.703961 seconds, one observation each.
Processes ran sequentially on the shared Mac, without host isolation. The native
runtime's source/binary is unchanged from the restored baseline after the
rejected lifetime experiment.

These are configuration-level results. Native uses block32 INT8 weights/dynamic
activations, original vision weights expanded to FP32, a prefix-plus-128 KV ring
and 35-gram suppression over 128 tokens. llama uses Q8_0 decoder weights, the
same F16/F32 projector as prior runs, full attention and no equivalent n-gram
suppression. This protocol uses the matched newline prompt, which differs from
the native product default. The [upstream diagnostic](../2026-09-28-receipt-diagnostic/README.md) confirms
that the receipt failure depends on this newline: the native original default
recognizes it with zero CER in both BF16 and dynamic INT8.

All pages remain frozen and no kernel setting was tuned using these outputs.
This corpus is synthetic and does not establish production-document coverage.
EOS indicates protocol completion, not text accuracy. Empty responses have no
reported decode throughput or nonempty first-token timing. Native internal
first-token timing differs from llama's first streamed token arrival.

Reproduce the full audit from the repository root:

```sh
PYTHONPATH=. .venv/bin/python docs/benchmarks/2026-09-28-heldout-repeated/summarize.py
```

`comparison.json`, raw streams, JSON outputs, process resource records, exact
commands and binary/runner hashes are retained. Native BF16/weight-only repeated
runs, fresh-process comparisons, further tuning and final review remain outside
this completed two-configuration experiment.
