"""After the complete audit, verify all native outputs against the prior Q8 run."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CURRENT=ROOT/'powered-abba-1'
BASELINE=ROOT.parent/'2026-09-28-current-q8/powered-abba-1'


def read(path):
    return json.loads(path.read_text())


for root in [BASELINE,CURRENT]:
    summary=read(root/'repeated-comparison.json')
    assert summary['audited_requests']==96 and summary['all_eos']
expected={}
for image in ['english','chinese','table','dense']:
    paths=sorted((BASELINE/'repeated-0-native').glob(f'*-{image}.raw.json'))
    assert len(paths)==6
    outputs=[read(path)['result'] for path in paths]
    first=outputs[0]
    assert all(row['text']==first['text'] and row['token_ids']==first['token_ids'] for row in outputs)
    expected[image]=first
counts={image:0 for image in expected}
for phase in [0,3]:
    directory=CURRENT/f'repeated-{phase}-native'
    process=read(directory/'process.json')
    assert process['binary_sha256']=='1f1902e4db8a0c9ad5ad79242a439c589ba423ab8462cf52754e8af285c1f193'
    assert process['resources']['exit_code']==0 and not process['resources']['forced_kill']
    records=sorted(directory.glob('*.record.json'))
    assert len(records)==24
    for path in records:
        record=read(path)
        prefix=directory/f"{record['index']:03d}-{record['image_id']}"
        raw=read(prefix.with_suffix('.raw.json'))
        assert raw['status']=='ok'
        actual=raw['result']; baseline=expected[record['image_id']]
        for field in ['text','token_ids','finish_reason','image_tokens','precision']:
            assert actual[field]==baseline[field], (phase,record['index'],field)
        assert actual['finish_reason']=='eos'
        counts[record['image_id']]+=1
assert all(n==12 for n in counts.values())
result=dict(exact_native_requests=sum(counts.values()),counts=counts,
            baseline=str(BASELINE),current=str(CURRENT),
            compared_fields=['text','token_ids','finish_reason','image_tokens','precision'],
            baseline_identity_sha256=hashlib.sha256((BASELINE/'repeated-identity.json').read_bytes()).hexdigest(),
            current_identity_sha256=hashlib.sha256((CURRENT/'repeated-identity.json').read_bytes()).hexdigest())
(ROOT/'native-output-equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
