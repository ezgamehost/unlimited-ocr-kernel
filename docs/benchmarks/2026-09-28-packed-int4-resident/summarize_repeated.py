"""Audit all four completed ABBA phases and summarize quality alongside latency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.benchmark import completion_text_without_eos
from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(values)
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    return dict(n=len(values), median=statistics.median(values), p95_nearest_rank=values[math.ceil(.95 * len(values)) - 1], minimum=values[0], maximum=values[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    root = parser.parse_args().input_root
    identity = read(root / 'repeated-identity.json')
    phases = []
    all_rows = {'native': [], 'llama': []}
    decoded = {'native': {}, 'llama': {}}
    reference = None
    for phase, engine in enumerate(['native', 'llama', 'llama', 'native']):
        directory = root / f'repeated-{phase}-{engine}'
        process, summary = read(directory / 'process.json'), read(directory / 'summary.json')
        assert summary['complete'] and summary['protocol_complete']
        assert summary['expected_requests'] == summary['collected_requests'] == 24
        assert process['engine'] == engine and process['failure'] is None
        assert process['binary_sha256'] == identity['hashes'][process['command'][0]]
        assert process['resources']['exit_code'] in ([0] if engine == 'native' else [0, -15])
        assert not process['resources']['forced_kill'] and process['resources']['peak_rss_bytes'] > 0
        assert process['warmups_per_image'] == 1 and process['measured_per_image'] == 5
        assert process['seed'] == 42 and process['max_tokens'] == 8192
        common = {k: process[k] for k in ['schedule', 'manifest', 'thread_environment', 'expected_tokens_by_image', 'native_prompt']}
        if reference is None:
            reference = common
        assert common == reference
        if engine == 'llama':
            command = process['command']
            for option, value in [('-ngl','0'),('--device','none'),('-t','8'),('-tb','8')]:
                assert command[command.index(option) + 1] == value
            for flag in ['--no-mmproj-offload', '--no-op-offload', '--no-kv-offload']:
                assert flag in command
            props = read(directory / 'props.json')
        truths = {item['id']: item for item in process['manifest']['images']}
        paths = sorted(directory.glob('*.record.json'))
        assert len(paths) == 24
        rows = []
        for path, job in zip(paths, process['schedule']):
            row = read(path)
            assert all(row[k] == v for k,v in job.items())
            assert row['index'] == len(rows)
            assert row['status'] == 'ok' and row['finish_reason'] == 'eos'
            item = truths[row['image_id']]
            assert row['image_sha256'] == item['sha256']
            prefix = directory / f"{row['index']:03d}-{row['image_id']}"
            output = read(prefix.with_suffix('.decoded.json'))
            raw = read(prefix.with_suffix('.raw.json'))
            if engine == 'native':
                result = raw['result']
                assert raw['status'] == 'ok' and result['finish_reason'] == 'eos'
                assert result['precision'] == row['precision'] == 'int4_body_int8_head_dynamic_int8_activations'
                assert result['timings'] == row['timings']
                expected_text, expected_tokens = result['text'], result['token_ids']
            else:
                assert raw['stop_type'] == 'eos' and not raw['truncated']
                assert raw['timings'] == row['timings']
                assert row['cache_tokens'] == row['timings']['cache_n'] == 0
                assert row['timings']['prompt_n'] == 1518
                expected_text, expected_tokens = completion_text_without_eos(raw, props['eos_token'])
            assert output == dict(text=expected_text, token_ids=expected_tokens)
            assert row['emitted_tokens'] == len(expected_tokens)
            cer = character_error_rate(normalize_ocr(item['text']), normalize_ocr(expected_text))
            assert math.isclose(cer, row['cer'], rel_tol=0, abs_tol=1e-12)
            row = dict(row, phase=phase)
            rows.append(row)
            all_rows[engine].append(row)
            decoded[engine].setdefault(row['image_id'], []).append(output)
        per_image = {}
        for image in truths:
            selected = [r for r in rows if r['image_id'] == image]
            assert len(selected) == 6
            assert [r['iteration'] for r in selected if r['warmup']] == [-1]
            assert sorted(r['iteration'] for r in selected if not r['warmup']) == list(range(5))
            per_image[image] = distribution(r['wall_seconds'] for r in selected if not r['warmup'])
        phases.append(dict(phase=phase, engine=engine, resources=process['resources'], startup_wall_seconds=process['startup_wall_seconds'], per_image_wall=per_image, binary_sha256=process['binary_sha256']))
    engines = {}
    for engine, rows in all_rows.items():
        assert len(rows) == 48
        images = {}
        for image, outputs in decoded[engine].items():
            assert len(outputs) == 12
            selected = [r for r in rows if r['image_id'] == image and not r['warmup']]
            assert len(selected) == 10
            variants = {hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest() for out in outputs}
            images[image] = dict(cer=sorted({r['cer'] for r in rows if r['image_id'] == image}),
                emitted_tokens=sorted({len(out['token_ids']) for out in outputs}), output_variants=len(variants),
                wall_seconds=distribution(r['wall_seconds'] for r in selected),
                first_token_seconds=distribution(r['first_token_seconds'] for r in selected),
                decode_tokens_per_second=distribution(r['decode_steps']/r['decode_seconds'] for r in selected if r['decode_steps'] > 0 and r['decode_seconds'] > 0))
        engines[engine] = dict(images=images, phases=[p for p in phases if p['engine'] == engine])
    comparison = {}
    for image in decoded['native']:
        n, l = engines['native']['images'][image], engines['llama']['images'][image]
        comparison[image] = dict(median_speed_ratio_llama_over_native=l['wall_seconds']['median']/n['wall_seconds']['median'],
            cer_equal=n['cer'] == l['cer'], emitted_tokens_equal=n['emitted_tokens'] == l['emitted_tokens'],
            native_cer=n['cer'], llama_cer=l['cer'])
    result = dict(audited_requests=96, all_eos=True, engines=engines, comparison=comparison,
        notes=['ABBA process order; two warmups and ten measurements per image/engine.',
               'Nearest-rank p95 with ten measurements is the maximum; all outliers retained.',
               'Hybrid Q4/Q8-head vs llama Q8_0, differing vision storage and attention/repetition policies.',
               'CER and generation length differences are reported; these are system-level results.',
               'TTFT/decode clock boundaries differ by engine; raw timings retained.',
               'Startup and exact-child lifetime RSS are per-process observations, not per-page measurements.',
               'Shared M4 Pro; no universal speed or host-isolation claim.'])
    (root / 'repeated-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
