"""Independently unpack every real packed tensor and compare to the Q4 simulation."""
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from tools.fetch_model import sha256

ROOT=Path(__file__).resolve().parent
packed=Path('artifacts/unlimited-ocr-q4-block32.safetensors')
simulation=Path('artifacts/unlimited-ocr-q4-sim-in-q8.safetensors')
manifest=json.loads(packed.with_suffix('.json').read_text())
oracle=json.loads(simulation.with_suffix('.json').read_text())
assert manifest['format']=='uocr-q4-block32-v1'
assert manifest['storage_bits']==manifest['weight_bits']==4
assert manifest['source_sha256']==oracle['source_sha256']
assert sha256(packed)==manifest['output_sha256']
assert sha256(simulation)==oracle['output_sha256']
count=elements=0
with safe_open(packed,framework='np') as p, safe_open(simulation,framework='np') as s:
    names={n for n in p.keys() if not n.endswith('.q4_scales')}
    assert names=={n for n in s.keys() if not n.endswith('.q8_scales')}
    assert p.metadata()['storage_bits']==p.metadata()['weight_bits']=='4'
    for name in sorted(names):
        original=s.get_slice(name); rows,cols=original.get_shape();blocks=(cols+31)//32
        stored=p.get_slice(name)
        assert stored.get_shape()==[rows,blocks,16]
        for start in range(0,rows,1024):
            end=min(start+1024,rows)
            data=stored[start:end,:,:]
            assert data.dtype==np.uint8
            low=(data&15).astype(np.int16); high=(data>>4).astype(np.int16)
            low[low>=8]-=16;high[high>=8]-=16
            codes=np.concatenate((low,high),axis=2).reshape(end-start,blocks*32)
            assert np.all((codes>=-7)&(codes<=7)),name
            assert np.all(codes[:,cols:]==0),name
            assert np.array_equal(codes[:,:cols],original[start:end,:]),name
        a=p.get_tensor(name+'.q4_scales');b=s.get_tensor(name+'.q8_scales')
        assert a.dtype==b.dtype==np.float32
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),name
        count+=1;elements+=rows*cols
assert count==manifest['matrix_count']==oracle['matrix_count']
result=dict(manifest=manifest,oracle_sha256=oracle['output_sha256'],verified_matrices=count,
            verified_weight_codes=elements,codes_exact=True,scales_bitwise_equal=True,
            simulation_bytes=oracle['bytes'],packed_bytes=manifest['bytes'],
            file_reduction_fraction=1-manifest['bytes']/oracle['bytes'],
            note='File representation parity only; packed runtime and performance are not yet verified.')
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
