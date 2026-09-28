# Rejected head scale-cache experiment

Interleaving the Q8 vocabulary head scales preserved all 16 fixed-work logit hashes, but did not improve measured decode time. The candidate was rejected and production source and the accepted CLI were restored.

The battery ABBA comparison used 512 teacher-forced steps, eight workers, one warmup and three measured repetitions per process (six measurements per variant). Decode including full-logit checksums had a baseline median of 3.6891910835 seconds and a candidate median of 3.6998895 seconds: 0.29% slower, with overlapping ranges. Prefill medians were 1.4936815205 and 1.492948458 seconds. The candidate additionally allocated 20,684,800 bytes for head scales. Battery results are exploratory, not an AC performance claim.

Focused tests passed, including bitwise comparisons, unsupported shapes, unaligned input and unchanged prefill. No full OCR acceptance run was performed for this nonwinning candidate. Raw records, identities and the rejected source patch are retained in `battery-abba-1/`.
