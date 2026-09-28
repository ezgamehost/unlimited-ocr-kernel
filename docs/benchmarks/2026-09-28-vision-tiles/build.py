"""Build isolated query-cap experiments, restoring source after each build."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path('src/cpu/ops.rs')
MARKER = 'let block = nq.min(128).min((cpu.scratch_limit / 4 - fixed) / nk);'


def main():
    original = SOURCE.read_text()
    assert original.count(MARKER) == 1
    env = dict(os.environ, RUSTUP_HOME=str(Path('.tools/rustup').resolve()),
               CARGO_HOME=str(Path('.tools/cargo').resolve()))
    command = ['.tools/cargo/bin/cargo', 'build', '--release', '--example', 'bench_vision_tiles']
    results = []
    for cap in [128, 256, 512, 1024]:
        target = Path(f'artifacts/vision-tiles/cap-{cap}')
        assert not target.exists(), target
        candidate = original.replace(MARKER, MARKER.replace('min(128)', f'min({cap})'))
        SOURCE.write_text(candidate)
        try:
            with (ROOT / f'build-{cap}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            shutil.copy2('target/release/examples/bench_vision_tiles', target)
            results.append(dict(cap=cap, binary=str(target.resolve()),
                                binary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                ops_sha256=hashlib.sha256(candidate.encode()).hexdigest()))
            print(json.dumps(results[-1]), flush=True)
        finally:
            assert SOURCE.read_text() == candidate, 'source changed concurrently; refusing overwrite'
            SOURCE.write_text(original)
    (ROOT / 'builds.json').write_text(json.dumps(dict(
        command=command, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        original_ops_sha256=hashlib.sha256(original.encode()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path('examples/bench_vision_tiles.rs').read_bytes()).hexdigest(),
        variants=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
