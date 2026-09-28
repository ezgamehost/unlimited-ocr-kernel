"""Diagnostic Q4-in-I8 overlay with only lm_head retained at Q8; no speed claim."""
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from safetensors import safe_open
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent
OUTPUT = Path('artifacts/unlimited-ocr-q4-q8head-sim.safetensors')
SOURCES = [Path('artifacts/unlimited-ocr-q4-sim-in-q8.safetensors'), Path('artifacts/unlimited-ocr-q8-block32.safetensors')]


def header(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        return json.loads(stream.read(size)), size + 8


def main():
    assert not OUTPUT.exists() and not OUTPUT.with_suffix('.json').exists()
    manifests = [json.loads(p.with_suffix('.json').read_text()) for p in SOURCES]
    for path, manifest in zip(SOURCES, manifests):
        assert sha256(path) == manifest['output_sha256']
    assert manifests[0]['source_sha256'] == manifests[1]['source_sha256']
    h4, start4 = header(SOURCES[0])
    h8, start8 = header(SOURCES[1])
    h4['__metadata__'].pop('simulated_weight_bits')
    h4['__metadata__']['experiment'] = 'q4_levels_except_q8_lm_head_in_i8_storage'
    raw = json.dumps(h4, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    start = len(raw) + 8
    fd, temporary = tempfile.mkstemp(dir=OUTPUT.parent, prefix=OUTPUT.name, suffix='.tmp')
    os.close(fd)
    try:
        with open(temporary, 'wb') as out, SOURCES[0].open('rb') as source:
            out.write(struct.pack('<Q', len(raw))); out.write(raw)
            source.seek(start4); shutil.copyfileobj(source, out, 1 << 20)
            with SOURCES[1].open('rb') as head:
                for name in ['lm_head.weight', 'lm_head.weight.q8_scales']:
                    assert h4[name]['dtype'] == h8[name]['dtype'] and h4[name]['shape'] == h8[name]['shape']
                    a, b = h8[name]['data_offsets']
                    c, d = h4[name]['data_offsets']
                    assert b - a == d - c
                    head.seek(start8 + a); out.seek(start + c)
                    remaining = b - a
                    while remaining:
                        chunk = head.read(min(1 << 20, remaining))
                        assert chunk
                        out.write(chunk); remaining -= len(chunk)
            out.flush(); os.fsync(out.fileno())
        with safe_open(temporary, framework='np') as check:
            assert len(check.keys()) == manifests[0]['matrix_count'] * 2
        # Independently verify each output tensor against its intended source.
        audit = {}
        with open(temporary, 'rb') as candidate:
            for path, reference_header, reference_start, is_head in [(SOURCES[0], h4, start4, False), (SOURCES[1], h8, start8, True)]:
                with path.open('rb') as source:
                    count = 0
                    for name, descriptor in h4.items():
                        if name == '__metadata__' or name.startswith('lm_head.weight') != is_head:
                            continue
                        a, b = descriptor['data_offsets']; c, d = reference_header[name]['data_offsets']
                        assert b - a == d - c
                        candidate.seek(start + a); source.seek(reference_start + c)
                        remaining = b - a
                        while remaining:
                            n = min(1 << 20, remaining)
                            assert candidate.read(n) == source.read(n), name
                            remaining -= n
                        count += 1
                    audit[str(path)] = count
        for path, manifest in zip(SOURCES, manifests):
            assert sha256(path) == manifest['output_sha256']
        manifest = dict(schema_version=1, format='uocr-q8-block32-v1', block_size=32,
                        source_sha256=manifests[0]['source_sha256'], output_sha256=sha256(Path(temporary)),
                        matrix_count=manifests[0]['matrix_count'], bytes=Path(temporary).stat().st_size,
                        storage_bits=8, experiment='Q4 levels except Q8 lm_head, all physical I8; no speedup claim',
                        input_manifests=manifests, verified_tensor_counts=audit)
        os.link(temporary, OUTPUT)
        OUTPUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (ROOT / 'head-simulation-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2))
    finally:
        os.unlink(temporary)


if __name__ == '__main__':
    main()
