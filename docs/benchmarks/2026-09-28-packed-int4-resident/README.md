# Full-page pilot catches an INT4 prompt sensitivity

All-Q4 did not pass the matched-prompt quality gate, so the planned repeated
llama.cpp comparison has not run. Its four-page pilot reached EOS everywhere,
but the table returned an empty string (zero emitted tokens, CER1). English,
Chinese and dense pages had CER0. The original-prompt table still works.
`pilot/` retains the exact requests, responses, timing and process resources.
An empty response is not counted as a performance win.

The benchmark's explicit `<image>\ndocument parsing.` prompt matches llama MTMD;
the runtime default remains `<image>document parsing.`. Both are tested here.
The same extra newline previously exposed an upstream receipt sensitivity,
but this table result is a distinct Q4 precision regression relative to Q8.

## Isolating the output head

The ignored `table_prompt_quantization_diagnostic` test computes Q4 decoder
hidden states and changes only the head used to obtain first-token logits.
All six reported rows match exactly between packed Q4 and the I8 simulation,
including the failing matched prompt. Thus this observed failure is not a
packed-kernel arithmetic mismatch. Reusing the same matched-prompt hidden state:

| Output head | EOS logit | det logit | Selected |
| --- | ---: | ---: | --- |
| Q4 + dynamic I8 inputs | 18.831995 | 18.223175 | EOS |
| Q8 + dynamic I8 inputs | 17.983084 | 18.106924 | det |
| Original BF16 + FP32 inputs | 18.047201 | 18.051664 | det |

The BF16 margin is especially small. These head substitutions identify a
sufficient first-token repair for this case, not universal prompt robustness.
The original prompt gives a much larger det margin. Full logs and diagnostic
executable hash are retained in `head-diagnostic-comparison.json`.

## Complete hybrid quality simulation

`prepare_head_simulation.py` constructs a physical I8 overlay from the verified
Q4 simulation, replacing only `lm_head.weight` and its scales with the verified
Q8 tensors. It byte-checks all 4,394 tensors against the intended source and
rechecks both source hashes before publishing. No original tensor is modified.
Artifact SHA256: `741b40e3adf06ad2abb05c593e13a184303fd7d820142de15c8be6d268175f6b`.
The full manifest and provenance are in `head-simulation-manifest.json`.

`head_quality.py` then runs all eight synthetic pages with the original prompt,
plus all four primary pages with the matched prompt. **All twelve requests
reach EOS and have CER0**, including both table prompts. The matched table emits
247 tokens. Original-prompt dense emits 1578 tokens versus 768 in all-Q4, while
matched dense emits 768; layout/grouping and generation length must stay visible
in subsequent timing comparisons. This is not token equivalence to all-Q4.

Raw responses, requests, executable hash and normal-exit resources are under
`head-simulation-quality/`. These pages informed precision selection and are not
unseen validation. The simulation still stores every code as I8, so it makes no
hybrid packing or performance claim. Next: explicit packed-Q4/Q8-head format,
strict loader and exact simulation parity, then resume full-page benchmarking.

Validation after adding the diagnostic: Rust suite 56 passed, 9 ignored;
clippy `--all-targets -- -D warnings` clean. The new ignored diagnostic was run
explicitly twice, once per packed/simulation overlay, and both runs passed.
No production inference code changed during this investigation.
