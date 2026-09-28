"""Audit the complete ABBA lifetime experiment; preserve per-process results."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('base-a', 'candidate-a', 'candidate-b', 'base-b')


def read(path):
    return json.loads(path.read_text())


def main():
    result, outputs, reference = {}, {}, None
    identity = read(ROOT / 'identity.json')
    for name in NAMES:
        directory = ROOT / name
        process, summary = read(directory / 'process.json'), read(directory / 'summary.json')
        assert summary['complete'] and summary['collected_requests'] == summary['expected_requests'] == 8
        assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
        assert process['resources']['peak_rss_bytes'] > 0 and process['failure'] is None
        variant = name.rsplit('-', 1)[0]
        assert process['binary_sha256'] == identity['binaries'][variant]
        if reference is None:
            reference = process
        for field in ('manifest', 'schedule', 'thread_environment', 'max_tokens', 'native_prompt'):
            assert process[field] == reference[field], field
        rows = [read(p) for p in sorted(directory.glob('*.record.json'))]
        assert len(rows) == 8
        assert all(r['status'] == 'ok' and r['finish_reason'] == 'eos' for r in rows)
        result[name] = dict(resources=process['resources'], startup_seconds=process['startup_wall_seconds'], images={})
        for item in process['manifest']['images']:
            image = item['id']
            selected = [r for r in rows if r['image_id'] == image]
            assert sorted(r['iteration'] for r in selected) == [-1, 0]
            assert sum(r['warmup'] for r in selected) == 1
            for r in selected:
                decoded = read(directory / f"{r['index']:03d}-{image}.decoded.json")
                if image not in outputs:
                    outputs[image] = decoded
                assert decoded == outputs[image], (name, image, 'output differs')
            measured = next(r for r in selected if not r['warmup'])
            result[name]['images'][image] = {k: measured[k] for k in ('wall_seconds', 'cer', 'emitted_tokens')}
    comparison = {}
    for image in outputs:
        baseline = [result[n]['images'][image]['wall_seconds'] for n in ('base-a', 'base-b')]
        candidate = [result[n]['images'][image]['wall_seconds'] for n in ('candidate-a', 'candidate-b')]
        comparison[image] = dict(baseline_seconds=baseline, candidate_seconds=candidate,
                                baseline_median=statistics.median(baseline),
                                candidate_median=statistics.median(candidate), exact_outputs_match=True)
    document = dict(processes=result, comparison=comparison,
                    notes=['Exploratory ABBA; two measured requests per image per variant, not final ten-repeat acceptance.',
                           'All 32 outputs, including warmups, match exactly by image.',
                           'Peak RSS is process-wide, including load and all eight requests.',
                           'Shared host; retain both order brackets and all timing variation.'])
    (ROOT / 'comparison.json').write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps(document, indent=2))


if __name__ == '__main__':
    main()
