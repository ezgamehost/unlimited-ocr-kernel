"""Compare head-major KV cache candidate against accepted token-major cache."""
import argparse
import json
import re
from pathlib import Path
import shutil
import statistics
import subprocess

from tools.benchmark import benchmark_environment
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def check_power(source):
    power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
    expected = 'AC Power' if source == 'ac' else 'Battery Power'
    assert f"Now drawing from '{expected}'" in power
    profile = subprocess.check_output(['pmset', '-g', 'custom'], text=True)
    if source == 'battery':
        charge = re.search(r'\t(\d+)%;', power)
        assert charge and int(charge.group(1)) >= 50, 'battery probe requires at least 50%'
        modes = re.findall(r'^\s*powermode\s+(\d+)\s*$', profile, re.MULTILINE)
        assert modes and all(mode == '0' for mode in modes), 'normal power mode required'
    return dict(battery=power, profile=profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', default='abba')
    parser.add_argument('--prepared', type=Path, help='Reuse binaries from an existing identity JSON')
    parser.add_argument('--power-source', choices=['ac', 'battery'], default='ac')
    args = parser.parse_args()
    check_power(args.power_source)
    output = ROOT / args.output_name
    output.mkdir(exist_ok=False)
    if args.prepared:
        prepared = json.loads(args.prepared.read_text())
        binaries = prepared['binaries']
        assert set(binaries) == {'baseline', 'packed'}
        for value in binaries.values():
            assert sha256(Path(value['binary'])) == value['sha256']
        save(output / 'prepared-identity.json', prepared)
    else:
        baseline = json.loads((ROOT.parent / '2026-09-28-q8-scheduling/abba/identity.json').read_text())['binaries']['baseline']
        assert sha256(Path(baseline['binary'])) == baseline['sha256']
        log = (ROOT / 'tests.log').read_text()
        executable = log.split('Running unittests src/lib.rs (')[1].split(')')[0]
        candidate = Path('artifacts/kv-layout/candidate')
        candidate.parent.mkdir(exist_ok=True)
        assert not candidate.exists()
        shutil.copy2(executable, candidate)
        binaries = dict(baseline=baseline, packed=dict(binary=str(candidate.resolve()), sha256=sha256(candidate),
            source_sha256={str(p): sha256(p) for p in map(Path, ['src/cpu/ops.rs', 'src/decoder/mod.rs', 'src/decoder/cache.rs'])}))
        (output / 'candidate.diff').write_text(subprocess.check_output(['git', 'diff', '--', 'src', 'tests/cache.rs', 'tests/decode_attention.rs'], text=True))
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-block32.safetensors')
    assert sha256(overlay) == '6981387fed4849dca9a713a717e9b54c04344adbaeff05410528773912e287ae'
    save(output / 'identity.json', dict(binaries=binaries, overlay=str(overlay), overlay_sha256=sha256(overlay),
                                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    reports = []
    for phase, name in enumerate(['baseline', 'packed', 'packed', 'baseline']):
        power = check_power(args.power_source)
        binary = binaries[name]
        assert sha256(Path(binary['binary'])) == binary['sha256']
        command = [binary['binary'], 'model::benchmark::fixed_work_decoder_benchmark',
                   '--ignored', '--exact', '--nocapture', '--test-threads=1']
        env = benchmark_environment(8, 8)
        env.update(UOCR_BENCH_WEIGHTS=str(overlay), UOCR_BENCH_THREADS='8')
        stem = output / f'{phase}-{name}'
        with stem.with_suffix('.log').open('wb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, start_new_session=True)
            resources = wait_child(child, 900)
        row = dict(phase=phase, variant=name, command=command, resources=resources, power=power,
                   power_source=args.power_source)
        row['samples'] = [json.loads(line[line.index('{'):]) for line in stem.with_suffix('.log').read_text().splitlines()
                          if '"benchmark":"native_fixed_work_decoder"' in line]
        save(stem.with_suffix('.json'), row)
        row['power_after'] = check_power(args.power_source)
        save(stem.with_suffix('.json'), row)
        assert resources['exit_code'] == 0 and not resources['forced_kill']
        assert len(row['samples']) == 4
        for i, sample in enumerate(row['samples']):
            assert sample['repetition'] == i and sample['warmup'] == (i == 0)
            assert sample['decode_steps'] == 512 and sample['prefix_tokens'] == 1517 and sample['threads'] == 8
        reports.append(row)
        print(json.dumps(dict(phase=phase, variant=name, decode=[s['decode_with_checksum_seconds'] for s in row['samples']])), flush=True)
    hashes = {s['logits_sha256'] for r in reports for s in r['samples']}
    assert hashes == {'c129d845d9d0f6b2b1f7bc1aaa7741d1a9fb95b428a1016f519df94acb862f3a'}
    summary = {}
    for name in binaries:
        rows = [s for r in reports if r['variant'] == name for s in r['samples'] if not s['warmup']]
        summary[name] = {metric: dict(median=statistics.median(s[metric] for s in rows),
                                     minimum=min(s[metric] for s in rows), maximum=max(s[metric] for s in rows))
                         for metric in ['prefill_seconds', 'decode_with_checksum_seconds']}
    save(output / 'summary.json', dict(variants=summary, logits_sha256=next(iter(hashes)), measured_samples_per_variant=6,
         power_source=args.power_source, note='Battery series is exploratory and separate from AC comparisons.' if args.power_source == 'battery' else 'AC series.'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
