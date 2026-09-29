# Shared quantized activation for concurrent Q/K/V

Status: candidate; no speed or acceptance claim yet.

QuantizedActivation owns one quantized row and its rank. Because input Tensor
validation and the single-row condition force all leading dimensions to one,
rank metadata preserves 1D/2D/3D output shapes without a new shape allocation.
Q8 and packed Q4 linear_prepared methods validate width and retain the existing
conservative scratch allowance. Ordinary dynamic projection methods retain their
pre-allocation shape/budget checks and delegate to the same arithmetic.

The concurrent decoder Q/K/V branch prepares one immutable row only when all three
projections use dynamic quantization. It shares the row across existing workers,
eliminating two redundant quantizations. The existing aggregate scratch gate and
serial/low-budget/weight-only/unquantized/partial-overlay fallbacks remain intact.

The missing-API RED test is retained. Operator tests cover scalar/accelerated paths,
Q8/Q4, widths 1/31/32/33/1280, ranks one through three, invalid width, multirow
inputs and nonfinite values. Standard Rust tests, formatting, lint, and existing
expert/decoder reference gates pass. The initial full Q8 digest matches the
accepted runtime in all four repetitions; a final digest run checks the subsequent
shape-Vec-to-rank metadata refinement independently.

Reviewer prepared_activation_review found no blocking correctness, resource, or
parallel-use issues. The reviewer noted prepared-versus-dynamic comparisons now
share implementation, making the independent reference and model-digest checks
essential. Review was static; no extra timing jobs ran concurrently.

The final metadata-refined build also matches the accepted full-logit digest in
all four repetitions. All twelve Q8 and twelve hybrid OCR outputs match exactly,
including coordinates, token IDs, EOS, image counts and precision; both child
processes exit normally without forced kills. Tested CLI SHA256:
a1af66f68ffa796e9a531d2fbd6fcd13428589d0548d6ee24601afee389b77ba.

## Battery ABBA result: no established gain

All four processes complete with unchanged battery source, normal power modes,
charge above 50%, and identical full-logit hashes across all sixteen repetitions.
One warmup per process is designated before running; all remaining samples are
retained. Six measured samples per variant give:

| Metric | Baseline | Shared Q/K/V |
|---|---:|---:|
| Decode median seconds | 3.184588 | 3.195385 |
| Decode min–max seconds | 3.158957–3.241361 | 3.177134–3.238854 |
| Prefill median seconds | 1.352769 | 1.348940 |

Decode median is about 0.34% higher and ranges overlap. This battery-only result
does not establish a gain or a reliable regression; it does not change the AC
headline. The Q/K/V-only candidate is not accepted as a speed optimization.

The complete implementation is preserved in source.patch and remains uncommitted
for the next experiment: sharing the same prepared row across the selected MoE
gate/up projections, which currently repeat its quantization fourteen times per
layer (six routed experts plus one shared expert, two projections each). That
broader experiment must independently establish output equivalence and performance.
