"""ABBA full-page resident comparison, gated on hybrid quality and logit parity."""
import argparse
import json
import subprocess
from pathlib import Path

from tools.benchmark_resident import run
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def main():
    gate = ROOT.parent / '2026-09-28-hybrid-int4'
    quality = json.loads((gate / 'packed-equivalence/comparison.json').read_text())
    assert len(quality) == 12 and all(all(row[key] for key in ['exact_text', 'exact_tokens', 'eos', 'image_tokens_equal']) for row in quality)
    assert json.loads((gate / 'fixed-work/summary.json').read_text())['logits_sha256']
    native = Path('target/release/uocr')
    llama = Path('/tmp/unlimited-ocr-baseline/bin/llama-b11223/llama-server')
    model = Path('/tmp/unlimited-ocr-baseline/models/model.gguf')
    projector = Path('artifacts/mmproj-unlimited-ocr-F16-tiles32.gguf')
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-block32.safetensors')
    overlay_manifest = json.loads(overlay.with_suffix('.json').read_text())
    expected = {llama: '6a072f8bf308ea28144476441d25760ee01b84ed102f86be4662d21ace8d72a6',
                model: '149574e63ec2681bbfb3d0f83ca1064e041fd6f3a97a62382d3168ad64289848',
                projector: '75dcbeafd883b3f27bee1a3d95c3ef57a6305c16758912af7cf62e6cf4c009a6',
                overlay: overlay_manifest['output_sha256']}
    hashes = {str(path.resolve()): sha256(path) for path in [native, *expected]}
    for path, digest in expected.items():
        assert hashes[str(path.resolve())] == digest, path
    assert hashes[str(native.resolve())] == json.loads((gate / 'packed-equivalence/process.json').read_text())['binary_sha256']
    identity = dict(hashes=hashes, overlay_manifest=overlay_manifest,
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    note='Identity verification precedes all inference timing; resident independent requests, no cold-cache claim.')
    (ROOT / 'repeated-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    for index, engine in enumerate(['native', 'llama', 'llama', 'native']):
        output = ROOT / f'repeated-{index}-{engine}'
        args = argparse.Namespace(engine=engine, binary=native if engine == 'native' else llama,
            model=Path('artifacts/upstream') if engine == 'native' else model,
            mmproj=None if engine == 'native' else projector,
            int8_weights=overlay if engine == 'native' else None,
            int8_activations=engine == 'native', manifest=Path('benchmarks/corpus/manifest.json'),
            output=output, threads=8, blas_threads=8, warmups=1, repetitions=5,
            seed=42, max_tokens=8192, timeout=180., startup_timeout=120.)
        print(json.dumps(dict(event='phase_start', phase=index, engine=engine, output=str(output))), flush=True)
        summary = run(args)
        assert summary['complete'] and summary['collected_requests'] == 24
        print(json.dumps(dict(event='phase_complete', phase=index, engine=engine)), flush=True)


if __name__ == '__main__':
    main()
