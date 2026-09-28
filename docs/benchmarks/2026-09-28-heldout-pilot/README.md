# Frozen-page correctness pilot

Each configuration processed the four newly frozen synthetic pages once, with
no warm-ups, at eight worker/BLAS threads and an 8,192-token cap. All processes
exited normally; all requests reached EOS with their independently declared
image/prompt counts. These are single observations for correctness triage, not
repeated latency results. The native binary is unchanged by this pilot.

| Page | Native dynamic INT8 CER | Native BF16 CER | llama Q8 CER |
| --- | ---: | ---: | ---: |
| Bilingual notice | 0 | 0 | 0 |
| Wide report | 0 | 0 | 0 |
| Archive register | 0 | 0 | 0.438795337 |
| Receipt | 1 | 1 | 1 |

**The receipt fails in all three configurations:** each returns empty text and
immediate EOS. It is not a successfully recognized page. This rules out dynamic
INT8 quantization alone as the cause, but an original-upstream inference check
is still required before attributing the failure to the model itself. The
receipt remains in the frozen corpus; no text or image is changed to improve
this result.

On the archive, native INT8 emits 1,790 tokens and reproduces the transcript.
llama emits 2,497 tokens with repeated/reordered content. These are runtime
configuration observations, not isolated kernel-accuracy claims: native retains
the prefix plus a 128-token generated-history ring and uses the official
35-gram suppression over 128 tokens; this llama configuration uses full
attention and does not implement that same suppression policy. Weight formats
also differ. All raw outputs remain available for inspection.

`comparison.json` retains one-shot timing, output counts, CER and process
resources. The native INT8, native BF16 and llama Q8 runs are stored under
`native/`, `native-bf16/` and `llama/`, respectively. All use the same matched
newline prompt. Three pages have zero text CER for native in both precisions;
that does not imply their raw layout coordinates or labels are identical.

The source corpus and generation protocol are in
[benchmarks/heldout](../../../benchmarks/heldout/README.md). Subsequent repeated
runs must keep every page, including the empty-output failure, and report
quality alongside timing. EOS-only completion summaries are protocol results,
not declarations of OCR correctness.
