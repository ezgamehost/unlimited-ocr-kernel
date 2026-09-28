"""Order-reversed native Q4 packed versus Q4-in-I8 fixed-work comparison."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from tools.benchmark import benchmark_environment
from tools.benchmark_resident import wait_child

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    directory = ROOT / 'fixed-work'
    directory.mkdir(exist_ok=False)
    build = (ROOT / 'fixed-work-build.log').read_text()
    binary = Path(build.split('Executable unittests src/lib.rs (')[1].split(')')[0]).resolve()
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    variants = {'simulation': 'artifacts/unlimited-ocr-q4-sim-in-q8.safetensors',
                'packed': 'artifacts/unlimited-ocr-q4-block32.safetensors'}
    reports = []
    for index, name in enumerate(['simulation', 'packed', 'packed', 'simulation']):
        env = benchmark_environment(8, 8)
        env['UOCR_BENCH_WEIGHTS'] = variants[name]
        env['UOCR_BENCH_THREADS'] = '8'
        command = [str(binary), 'model::benchmark::fixed_work_decoder_benchmark', '--ignored', '--exact', '--nocapture', '--test-threads=1']
        row = dict(index=index, variant=name, command=command, binary_sha256=binary_hash,
                   overlay_manifest=json.loads(Path(variants[name]).with_suffix('.json').read_text()))
        save(directory / f'{index}-{name}.json', row)
        start = time.monotonic()
        with (directory / f'{index}-{name}.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            row['resources'] = wait_child(child, 900)
        row['wall_seconds'] = time.monotonic() - start
        row['samples'] = [json.loads(line[line.index('{'):]) for line in (directory / f'{index}-{name}.log').read_text().splitlines() if '"benchmark":"native_fixed_work_decoder"' in line]
        save(directory / f'{index}-{name}.json', row)
        assert row['resources']['exit_code'] == 0 and not row['resources']['forced_kill']
        assert len(row['samples']) == 4
        reports.append(row)
        print(json.dumps(dict(index=index, variant=name, wall=row['wall_seconds'], peak=row['resources']['peak_rss_bytes'])), flush=True)
    hashes = {sample['logits_sha256'] for row in reports for sample in row['samples']}
    assert len(hashes) == 1, hashes
    results = {}
    for name in variants:
        samples = [sample for row in reports if row['variant'] == name for sample in row['samples'] if not sample['warmup']]
        results[name] = {metric: statistics.median(s[metric] for s in samples) for metric in ['prefill_seconds', 'decode_with_checksum_seconds']}
    save(directory / 'summary.json', dict(logits_sha256=next(iter(hashes)), measured_samples_per_variant=6,
        medians=results, ratios_simulation_over_packed={metric: results['simulation'][metric] / results['packed'][metric] for metric in results['packed']},
        note='512 identical teacher-forced steps, including additive checksum and SHA256; original 1517-token prompt prefix; no vision timing; no llama comparison.'))


if __name__ == '__main__':
    main()
