"""Classify, without accepting, output differences from the RMSNorm candidate."""
import json
from pathlib import Path
import re

from tools.score import normalize_ocr

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / '2026-09-28-current-q8/quality'
CANDIDATE = ROOT / 'quality/q8'
PATTERN = re.compile(r'(<\|det\|>[^<\n]*?)\[(\d+), (\d+), (\d+), (\d+)\](<\|/det\|>)')


def read(path):
    return [json.loads(line)['result'] for line in path.read_text().splitlines()]


def main():
    before = read(BASELINE / 'responses.jsonl')
    after = read(CANDIDATE / 'responses.jsonl')
    cases = json.loads((BASELINE / 'process.json').read_text())['cases']
    assert len(before) == len(after) == len(cases) == 12
    results = []
    for case, old, new in zip(cases, before, after):
        mask = lambda s: PATTERN.sub(lambda m: m[1] + '[BOX]' + m[6], s)
        a, b = list(PATTERN.finditer(old['text'])), list(PATTERN.finditer(new['text']))
        assert mask(old['text']) == mask(new['text']) and len(a) == len(b)
        changes = []
        for index, (left, right) in enumerate(zip(a,b)):
            x = [int(left[i]) for i in range(2,6)]
            y = [int(right[i]) for i in range(2,6)]
            if x != y:
                changes.append(dict(box=index, before=x, after=y,
                                    maximum_coordinate_delta=max(abs(v-w) for v,w in zip(x,y))))
        results.append(dict(case=case['case'], boxes=len(a), changed_boxes=len(changes),
            max_coordinate_delta=max((v['maximum_coordinate_delta'] for v in changes),default=0),
            normalized_text_equal=normalize_ocr(old['text'])==normalize_ocr(new['text']),
            equal_token_count=len(old['token_ids'])==len(new['token_ids']),
            eos=old['finish_reason']==new['finish_reason']=='eos', changes=changes))
    (ROOT / 'coordinate-differences.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict((k,v) for k,v in row.items() if k != 'changes') for row in results],indent=2))


if __name__ == '__main__':
    main()
