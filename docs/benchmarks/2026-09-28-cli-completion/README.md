# Conservative CLI completion classification

The real maximum-crop diagnostic logged 255 decoder evaluations at an output
cap of 256. The first generated token can come from prefill, so this does not
prove EOS. The old heuristic incorrectly admitted this boundary as completed.

`run_llama` now labels `eval_runs == token_cap - 1` as unknown. The existing
aggregator excludes unknown completions from successful latency statistics.
Counts at or above the cap remain token_limit; missing timing remains unknown.
The cap-one case is covered too. Conservative exclusion may also reject a real
EOS at this boundary; use the resident server's explicit stop metadata when
an unambiguous finish reason is required.

Tests replay the retained real log through a child process and exercise the
boundary counts. They fail before the change and pass after it. The full Python
suite runs 29 tests: 27 pass, two artifact-dependent tests skip. Audit of the five
versioned JSON records containing llama-mtmd-cli commands and decode_steps finds
no remaining EOS claim at this exact boundary; the maximum-crop diagnostic was
already marked unknown. audit.json states the exact audit scope. Other stop
causes are not independently revalidated by that audit.

The repeated resident comparison uses explicit server stop metadata and is
unaffected. This change makes no inference-performance claim.
