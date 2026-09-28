"""Disposable Q8 scheduling probe; restore production source before timing."""
import hashlib
import json
import os
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


def main():
    output = ROOT / 'abba'
    output.mkdir(exist_ok=False)
    source = Path('src/cpu/quantized.rs')
    original = source.read_bytes()
    old = b'.with_min_len(self.rows.div_ceil(4).div_ceil(cpu.threads()))'
    new = b'.with_min_len(self.rows.div_ceil(4).div_ceil(cpu.threads()).div_ceil(4))'
    assert original.count(old) == 1
    environment = dict(os.environ, RUSTUP_HOME=str(Path('.tools/rustup').resolve()),
                       CARGO_HOME=str(Path('.tools/cargo').resolve()))
    binaries = {}
    try:
        for name, content in [('baseline', original), ('fine', original.replace(old, new))]:
            source.write_bytes(content)
            with (output / f'{name}-build.jsonl').open('w') as out, (output / f'{name}-build.stderr').open('w') as err:
                subprocess.run(['.tools/cargo/bin/cargo', 'test', '--release', '--lib', '--no-run',
                                '--message-format=json'], env=environment, stdout=out, stderr=err, check=True)
            events = [json.loads(line) for line in (output / f'{name}-build.jsonl').read_text().splitlines()]
            executable = next(row['executable'] for row in events if row.get('reason') == 'compiler-artifact'
                              and row.get('executable') and row['profile']['test'])
            binary = Path('artifacts/q8-scheduling') / name
            binary.parent.mkdir(exist_ok=True)
            assert not binary.exists()
            shutil.copy2(executable, binary)
            binaries[name] = dict(binary=str(binary.resolve()), sha256=sha256(binary),
                                  source_sha256=hashlib.sha256(content).hexdigest())
        (output / 'candidate.diff').write_text(subprocess.check_output(['git', 'diff', '--', str(source)], text=True))
    finally:
        source.write_bytes(original)
    overlay = Path('artifacts/unlimited-ocr-q4-q8head-block32.safetensors')
    assert sha256(overlay) == '6981387fed4849dca9a713a717e9b54c04344adbaeff05410528773912e287ae'
    save(output / 'identity.json', dict(binaries=binaries, overlay=str(overlay), overlay_sha256=sha256(overlay),
                                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    reports = []
    for phase, name in enumerate(['baseline', 'fine', 'fine', 'baseline']):
        power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        assert "Now drawing from 'AC Power'" in power
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
        row = dict(phase=phase, variant=name, command=command, resources=resources, power=power)
        row['samples'] = [json.loads(line[line.index('{'):]) for line in stem.with_suffix('.log').read_text().splitlines()
                          if '"benchmark":"native_fixed_work_decoder"' in line]
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
    save(output / 'summary.json', dict(variants=summary, logits_sha256=next(iter(hashes)), measured_samples_per_variant=6))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
