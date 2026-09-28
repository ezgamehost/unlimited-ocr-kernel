"""Audit full held-out runs; keep protocol completion distinct from OCR quality."""
import json
import math
import statistics
from pathlib import Path

from tools.score import character_error_rate, normalize_ocr

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    assert all(math.isfinite(v) and v >= 0 for v in values)
    return dict(samples=len(values), median=statistics.median(values),
                p95_nearest_rank=values[math.ceil(.95 * len(values)) - 1])


def engine(name):
    directory = ROOT / name
    summary, process = read(directory / 'summary.json'), read(directory / 'process.json')
    assert summary['complete'] and summary['protocol_complete']
    assert summary['collected_requests'] == summary['expected_requests'] == 48
    assert process['resources']['exit_code'] == 0 and not process['resources']['forced_kill']
    assert process['resources']['peak_rss_bytes'] > 0 and process['failure'] is None
    rows = [read(p) for p in sorted(directory.glob('*.record.json'))]
    assert len(rows) == 48 and sorted(r['index'] for r in rows) == list(range(48))
    for r in rows:
        assert r['status'] == 'ok' and r['finish_reason'] == 'eos'
        assert {k: r[k] for k in ('image_id', 'iteration', 'warmup')} == process['schedule'][r['index']]
        expected = process['expected_tokens_by_image'][r['image_id']]
        if name == 'llama':
            assert r['cache_tokens'] == 0 and r['timings']['prompt_n'] == expected['prefix_tokens']
        else:
            raw = read(directory / f"{r['index']:03d}-{r['image_id']}.raw.json")
            assert raw['result']['image_tokens'] == expected['image_tokens']
    result = dict(resources=process['resources'], startup_seconds=process['startup_wall_seconds'], images={})
    outputs = {}
    for item in process['manifest']['images']:
        image = item['id']
        selected = [r for r in rows if r['image_id'] == image]
        assert sorted(r['iteration'] for r in selected) == list(range(-2, 10))
        decoded = [read(directory / f"{r['index']:03d}-{image}.decoded.json") for r in selected]
        assert all(d == decoded[0] for d in decoded), (name, image, 'output instability')
        outputs[image] = decoded[0]
        cer = character_error_rate(normalize_ocr(item['text']), normalize_ocr(decoded[0]['text']))
        assert all(r['cer'] == cer and r['emitted_tokens'] == len(d['token_ids']) for r, d in zip(selected, decoded))
        measured = [r for r in selected if not r['warmup']]
        assert len(measured) == 10
        result['images'][image] = dict(cer=cer, emitted_tokens=len(decoded[0]['token_ids']),
            zero_cer_nonempty_requests=sum(r['cer'] == 0 and r['emitted_tokens'] > 0 for r in measured),
            empty_output_requests=sum(r['emitted_tokens'] == 0 for r in measured),
            wall_seconds=distribution(r['wall_seconds'] for r in measured),
            nonempty_first_token_seconds=distribution(r['first_token_seconds'] for r in measured if r['emitted_tokens'] > 0),
            decode_tokens_per_second=distribution(r['decode_steps'] / r['decode_seconds'] for r in measured
                                                  if r['decode_steps'] > 0 and r['decode_seconds'] > 0))
    return result, outputs, process


def main():
    native, no, np = engine('native')
    llama, lo, lp = engine('llama')
    for field in ('schedule', 'manifest', 'thread_environment', 'expected_tokens_by_image', 'max_tokens'):
        assert np[field] == lp[field], field
    comparison = {}
    for name, n in native['images'].items():
        l = llama['images'][name]
        eligible = n['zero_cer_nonempty_requests'] == l['zero_cer_nonempty_requests'] == 10
        comparison[name] = dict(both_zero_cer_nonempty=eligible,
            median_speed_ratio_llama_over_native=(l['wall_seconds']['median'] / n['wall_seconds']['median'] if eligible else None),
            exact_raw_output_match=no[name] == lo[name],
            note='Ratio withheld unless both configurations have nonempty zero-CER output on every measured request.')
    report = dict(native=native, llama=llama, comparison=comparison,
        notes=['All 48 requests per engine audited; 2 warmups and 10 measured requests per page.',
               'Nearest-rank p95 is the maximum of ten samples; no samples discarded.',
               'EOS-only summaries establish protocol completion, not OCR correctness.',
               'Empty outputs have no decode throughput or content-arrival estimate.',
               'Native internal TTFT differs from llama streamed-token arrival.',
               'Native prefix-plus-128 KV and 35-gram suppression differ from llama full attention without equivalent suppression.',
               'Weight formats differ; sequential runs on a shared Mac, not isolated kernel comparisons.'])
    (ROOT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
