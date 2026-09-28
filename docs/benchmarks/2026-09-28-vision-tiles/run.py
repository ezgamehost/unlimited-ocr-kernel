"""Measure isolated full-vision variants in forward/reverse process order."""
import datetime
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from tools.benchmark import benchmark_environment
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam-only', action='store_true')
    options = parser.parse_args()
    output = ROOT / ('sam-abba' if options.sam_only else 'pilot')
    assert not output.exists(), 'preserve existing measurements'
    builds = json.loads((ROOT / 'builds.json').read_text())
    variants = {row['cap']: row for row in builds['variants']}
    order = [128, 256, 512, 1024, 1024, 512, 256, 128]
    if options.sam_only:
        sam = json.loads((ROOT / 'sam1024-identity.json').read_text())
        variants = {128: variants[128], 1024: dict(sam, cap=1024, unbiased_query_cap=128)}
        order = [128, 1024, 1024, 128]
    for row in variants.values():
        assert sha256(Path(row['binary'])) == row['binary_sha256']
    power = subprocess.check_output(['pmset','-g','batt'], text=True)
    assert "Now drawing from 'AC Power'" in power
    model = Path('artifacts/upstream/model-00001-of-000001.safetensors')
    model_sha = sha256(model)
    assert model_sha == '2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6'
    output.mkdir()
    observations = []
    for phase, cap in enumerate(order):
        power = subprocess.check_output(['pmset','-g','batt'], text=True)
        assert "Now drawing from 'AC Power'" in power
        stem = output / f'{phase}-cap-{cap}'
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        started = time.monotonic()
        with stem.with_suffix('.stdout').open('w') as out, stem.with_suffix('.stderr').open('w') as err:
            completed = subprocess.run([variants[cap]['binary']], env=benchmark_environment(8,8),
                                       stdout=out, stderr=err, timeout=180)
        elapsed = time.monotonic() - started
        metadata = dict(phase=phase, cap=cap, utc=stamp, power=power,
                        variant=variants[cap],
                        binary_sha256=variants[cap]['binary_sha256'], model_sha256=model_sha,
                        exit_code=completed.returncode, process_wall_seconds=elapsed)
        stem.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        completed.check_returncode()
        rows = [json.loads(line) for line in stem.with_suffix('.stdout').read_text().splitlines()]
        assert len(rows) == 4
        for i, row in enumerate(rows):
            assert row['iteration'] == i and row['warmup'] == (i == 0)
            assert row['shape'] == [1513,1280]
            assert row['seconds'] > 0
        observations.extend(dict(row, cap=cap, phase=phase) for row in rows)
        print(json.dumps(dict(phase=phase, cap=cap,
                              measured_seconds=[r['seconds'] for r in rows if not r['warmup']],
                              hashes=sorted({r['sha256'] for r in rows}))), flush=True)
    summary = {}
    for cap in variants:
        rows = [r for r in observations if r['cap'] == cap]
        times = [r['seconds'] for r in rows if not r['warmup']]
        assert len(times) == 6 and len(rows) == 8
        summary[cap] = dict(median_seconds=statistics.median(times), minimum=min(times),
                            maximum=max(times), hashes=sorted({r['sha256'] for r in rows}))
    (output/'summary.json').write_text(json.dumps(dict(variants=summary, observations=observations),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
