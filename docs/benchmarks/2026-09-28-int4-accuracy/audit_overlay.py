"""Verify every simulated weight code and pin the exported overlay manifest."""
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from tools.fetch_model import sha256

ROOT=Path(__file__).resolve().parent
path=Path('artifacts/unlimited-ocr-q4-sim-in-q8.safetensors')
manifest=json.loads(path.with_suffix('.json').read_text())
assert manifest['simulated_weight_bits']==4 and manifest['storage_bits']==8
assert sha256(path)==manifest['output_sha256']
count=elements=0
minimum,maximum=0,0
with safe_open(path,framework='np') as reader:
    assert reader.metadata()['simulated_weight_bits']=='4' and reader.metadata()['storage_bits']=='8'
    for name in reader.keys():
        if name.endswith('.q8_scales'):
            continue
        value=reader.get_tensor(name)
        assert value.dtype==np.int8
        low,high=int(value.min()),int(value.max())
        assert -7<=low<=high<=7,name
        minimum=min(minimum,low);maximum=max(maximum,high)
        count+=1;elements+=value.size
assert count==manifest['matrix_count']
result=dict(manifest=manifest,verified_matrix_count=count,weight_elements=elements,
            minimum_code=minimum,maximum_code=maximum,all_codes_in_signed_int4_range=True,
            note='Codes stored as I8; no storage compression or performance claim.')
(ROOT/'overlay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
