"""Audit completed BF16 outputs against Q8 without implying a completed run."""
import json
from pathlib import Path

from tools.score import normalize_ocr

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    current = ROOT / 'llama-resident'
    previous = ROOT.parent / '2026-09-28-resident-repeated' / 'llama'
    process, baseline = read(current / 'process.json'), read(previous / 'process.json')
    assert process['manifest'] == baseline['manifest']
    assert process['schedule'] == baseline['schedule']
    records = [read(p) for p in sorted(current.glob('*.record.json'))]
    assert all(r['status'] == 'ok' and r['finish_reason'] == 'eos'
               and r['cache_tokens'] == 0 and r['timings']['prompt_n'] == 1518 for r in records)
    result = dict(completed_requests_audited=len(records),
                  expected_requests=len(process['schedule']), images={})
    for item in process['manifest']['images']:
        name = item['id']
        selected = [r for r in records if r['image_id'] == name]
        if not selected:
            continue
        values = [read(current / f"{r['index']:03d}-{name}.decoded.json") for r in selected]
        assert all(v == values[0] for v in values), f'BF16 output instability: {name}'
        old_records = [read(p) for p in previous.glob(f'*-{name}.record.json')]
        old = [read(previous / f"{r['index']:03d}-{name}.decoded.json") for r in old_records]
        assert old and all(v == old[0] for v in old)
        a, b = values[0], old[0]
        result['images'][name] = dict(
            requests_audited=len(selected), bf16_outputs_identical=True,
            bf16_emitted_tokens=len(a['token_ids']), q8_emitted_tokens=len(b['token_ids']),
            bf16_cer=sorted({r['cer'] for r in selected}),
            q8_cer=sorted({r['cer'] for r in old_records}),
            exact_raw_text_match=a['text'] == b['text'],
            normalized_transcript_match=normalize_ocr(a['text']) == normalize_ocr(b['text']),
            token_id_differences=sum(x != y for x, y in zip(a['token_ids'], b['token_ids']))
                                 + abs(len(a['token_ids']) - len(b['token_ids'])))
    summary_path = current / 'summary.json'
    result['benchmark_complete'] = (summary_path.exists() and read(summary_path)['complete']
                                    and len(records) == len(process['schedule']))
    result['notes'] = [
        'All completed requests are audited, including warmups; no timing estimates here.',
        'Normalized text equality does not imply matching layout labels or coordinates.',
        'This snapshot is partial until benchmark_complete is true; rerun after termination.']
    (ROOT / 'output-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
