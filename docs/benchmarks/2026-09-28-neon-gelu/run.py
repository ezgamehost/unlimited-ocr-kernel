"""Full-vision ABBA comparison against the accepted parallel-normalization executable."""
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from tools.benchmark import benchmark_environment
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'abba'
    assert not output.exists()
    baseline = json.loads((ROOT.parent / '2026-09-28-parallel-norm/abba/identity.json').read_text())['variants']['parallel']
    candidate = Path('artifacts/vision-gelu/neon-small')
    assert not candidate.exists()
    candidate.parent.mkdir(exist_ok=True)
    shutil.copy2('target/release/examples/bench_vision_tiles', candidate)
    variants = dict(baseline=baseline, neon=dict(binary=str(candidate.resolve()),
        binary_sha256=sha256(candidate), source_sha256={str(p): sha256(p) for p in
            map(Path, ['src/cpu/ops.rs', 'src/vision/mod.rs', 'src/vision/sam.rs',
                       'src/vision/clip.rs', 'src/cpu/mod.rs', 'src/cpu/neon_gelu.rs', 'examples/bench_vision_tiles.rs'])}))
    for value in variants.values():
        assert sha256(Path(value['binary'])) == value['binary_sha256']
    power = subprocess.check_output(['pmset','-g','batt'], text=True)
    assert "Now drawing from 'AC Power'" in power
    model_sha = sha256(Path('artifacts/upstream/model-00001-of-000001.safetensors'))
    assert model_sha == '2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6'
    output.mkdir()
    (output/'identity.json').write_text(json.dumps(dict(variants=variants, model_sha256=model_sha,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
    (output/'candidate.diff').write_text(subprocess.check_output(['git','diff','--','src'],text=True))
    (output/'candidate-neon_gelu.rs.txt').write_text(Path('src/cpu/neon_gelu.rs').read_text())
    observations = []
    for phase, label in enumerate(['baseline','neon','neon','baseline']):
        power = subprocess.check_output(['pmset','-g','batt'],text=True)
        assert "Now drawing from 'AC Power'" in power
        stem = output/f'{phase}-{label}'
        with stem.with_suffix('.stdout').open('w') as out, stem.with_suffix('.stderr').open('w') as err:
            result = subprocess.run([variants[label]['binary']], env=benchmark_environment(8,8),
                                    stdout=out, stderr=err, timeout=180)
        stem.with_suffix('.json').write_text(json.dumps(dict(power=power, exit_code=result.returncode),indent=2)+'\n')
        result.check_returncode()
        rows = [json.loads(line) for line in stem.with_suffix('.stdout').read_text().splitlines()]
        assert len(rows) == 4
        for index, row in enumerate(rows):
            assert row['iteration'] == index and row['warmup'] == (index == 0)
            assert row['shape'] == [1513,1280] and row['seconds'] > 0
        observations.extend(dict(row, variant=label, phase=phase) for row in rows)
        print(json.dumps(dict(phase=phase,variant=label,seconds=[r['seconds'] for r in rows],
                              hashes=sorted({r['sha256'] for r in rows}))),flush=True)
    summary = {}
    for label in variants:
        rows = [r for r in observations if r['variant'] == label and not r['warmup']]
        times = [r['seconds'] for r in rows]
        assert len(times) == 6
        summary[label] = dict(median_seconds=statistics.median(times),minimum=min(times),maximum=max(times))
    hashes = sorted({r['sha256'] for r in observations})
    (output/'summary.json').write_text(json.dumps(dict(variants=summary,embedding_hashes=hashes,
                                                     observations=observations),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    assert len(hashes) == 1, 'embedding parity failed'


if __name__ == '__main__':
    main()
