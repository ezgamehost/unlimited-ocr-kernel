# Native runtime pilot — not a final benchmark

Apple M4 Pro, original BF16 checkpoint, FP32 computation, CPU only. Native
Rust runtime uses Accelerate matrix multiplication and ARM64 NEON BF16 dot
products. Python/PyTorch is used only by the separate reference exporter.

Run the resident pilot after `cargo build --release --bin uocr`:

```sh
 target/release/uocr --jsonl
```

Send the following JSON line three times:

```json
{"image":"/tmp/unlimited-ocr-baseline/baidu.png","mode":"base","generation":{"max_tokens":128}}
```

`resident.jsonl` contains all three responses. The last two requests took
5.024 and 4.926 seconds inside inference, excluding model loading. All three
produced identical tokens and EOS. Vision consumed about 3.77 seconds on the
last two requests. These short outputs are unsuitable for sustained decode
throughput claims.

`scalar-dot-warm.json` records the earlier auto backend with scalar BF16 dots;
`neon-dot.json` records the first NEON run. Decode took 1.246 versus 0.600
seconds with identical output tokens. File residency differed, so the total
latencies of those two files do not isolate the kernel change. More repetitions
and controlled conditions are required before reporting a speedup.

The earlier llama.cpp Q8 baseline uses different precision, vision weights,
attention behavior, output tokens and fresh-process timing. This pilot does
not establish a win against it. Full document accuracy and repeated comparisons
remain pending.

## Vector softmax experiment

`vector-softmax-resident.jsonl` records five requests after adding vector
exponentials and NEON reductions, before the subsequent LayerNorm precision
fix. Last three inference times: 4.048, 4.023, 4.016 seconds; vision: 2.943,
2.922, 2.922 seconds. Output token IDs stayed identical to the earlier pilot.
This is preliminary sequential evidence, not a randomized comparative trial.

Reproduce the isolated operator comparison using
`cargo run --release --example bench_softmax`. The recorded run in
`softmax-microbench.log` processes 16 million elements per sample. At width
4096, the scalar path took 34.2–34.6 ms and the vector path 11.5–11.7 ms.
The implementation combines ARM64 maximum/sum reductions with Accelerate
`vvexpf`. Its in-place operation follows the
[Apple vForce contract](https://developer.apple.com/documentation/accelerate/vforce-library).

## Multi-crop accuracy regression

The original upstream export for the English page uses a 3×4 crop grid.
All normalized global/crop pixels match exactly. The first comparison exposed
an existing numerical error in wide, nearly constant LayerNorm rows: sequential
FP32 mean accumulation distorted centering. A 3072-element alternating
1.0/1.0001 regression reproduced the error independently of the model.

LayerNorm now retains FP64 precision through centering, variance and affine
calculation; stored activations remain FP32. Decoder RMSNorm is unchanged.
Both the logo and English global-stage tests pass, and the merged 12-crop
embedding passes the original `1e-4 + 1e-4*abs(reference)` tolerance. English
CLIP maximum error fell from 3.53e-4 to 5.38e-5. No tolerance was relaxed.

Generate and check the multi-crop reference:

```sh
python -m tools.reference --image benchmarks/corpus/english.png --mode gundam --vision --output artifacts/reference-english-gundam-f32
cargo test --release --test vision -- --include-ignored --test-threads=1
```
