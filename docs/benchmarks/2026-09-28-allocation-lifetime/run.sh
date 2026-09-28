#!/bin/sh
set -eu
# Run from repository root after producing the two archived executables.
for variant in base-a candidate-a candidate-b base-b; do
    .venv/bin/python -m tools.benchmark_resident \
        --engine native --binary "artifacts/uocr-lifetime-${variant%-*}" \
        --model artifacts/upstream \
        --int8-weights artifacts/unlimited-ocr-q8-block32.safetensors \
        --int8-activations --threads 8 --blas-threads 8 \
        --warmups 1 --repetitions 1 --max-tokens 8192 --timeout 300 \
        --output "docs/benchmarks/2026-09-28-allocation-lifetime/$variant"
done
