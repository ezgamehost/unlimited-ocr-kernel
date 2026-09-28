# Maximum 32-crop resource check and retained OCR failure

The deterministic 640×20,480 portrait has one numbered three-line text block in
each of 32 tiles. `generate.py` records the font, image hash and exact transcript.
The native run reports 3,793 image tokens, consistent with the admitted 1×32 crop
grid: 3,520 local tokens plus 273 global/separator tokens.

The dynamic INT8 runtime exits normally without a forced kill and reaches peak
child RSS 7,129,841,664 bytes (6.64 GiB), measured with wait4 for the exact PID.
This verifies resource behavior for this real maximum-crop input on the 24-GiB
M4 Pro. It does not establish a global process-memory bound or cover all image
aspect ratios. Scratch remains a checked-operator allocation limit.

**OCR fails.** The run emits an unrelated sequence of numbers, contains none of
the SECTION labels, and hits the 8,192-token cap. CER is 7.184860, including the
large insertion error. Wall time is 104.463 seconds including model load;
vision is 16.903 seconds, prefill 4.454 seconds, and decode 75.968 seconds. These
are failed-request resource measurements, not a completed-page latency win.

Two 256-token diagnostics investigate the onset rather than attempt complete
OCR. Native BF16 also outputs a number sequence (starting at 2). llama.cpp Q8
with the same F16 tiles32 projector also outputs a number sequence (starting at
1), with 3,798 prompt tokens. Thus dynamic INT8 rounding alone does not explain
the failure, and it reproduces across the two runtimes. This does not prove an
upstream model limitation: an original-reference check is still needed before
attributing the cause. Neither diagnostic is counted as successful OCR.

The legacy CLI parser initially labeled llama's 255 eval runs at a 256-token cap
as EOS. That inference is invalid because the first token comes from prefill.
The diagnostic record retains the old label for audit and marks the finish
reason unknown. The raw logs and output are unchanged. Fixing that heuristic is
pending; the completed resident comparison uses explicit server stop metadata
and is unaffected by this CLI-only ambiguity.

All commands, binary identity, thread settings, output, resource records and
input generation are retained here. Runtime binary SHA256 is unchanged from
the repeated resident comparison. No model/runtime code was changed for this
check. Worker sweep and maximum-crop results do not replace the pending BF16,
fresh-process, held-out tuning and public-input hardening checks.
