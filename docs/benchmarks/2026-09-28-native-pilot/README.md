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
