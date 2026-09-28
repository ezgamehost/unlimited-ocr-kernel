"""Prepare a bounded-memory, versioned INT8 decoder overlay; leave source intact."""
import argparse
import json
import os
import re
import struct
import tempfile
from pathlib import Path
import numpy as np
from safetensors import safe_open
from tools.fetch_model import sha256

BLOCK=32
FORMAT='uocr-q8-block32-v1'
SUFFIX='.q8_scales'
MATRIX=re.compile(r'^(?:lm_head|model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:(?:experts\.\d+|shared_experts)\.)?(?:gate|up|down)_proj))\.weight$')


def quantize_blocks(values):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('quantization needs a finite nonempty matrix')
    rows,cols=x.shape
    blocks=(cols+BLOCK-1)//BLOCK
    padded=np.zeros((rows,blocks*BLOCK),dtype=np.float32)
    padded[:,:cols]=x
    grouped=padded.reshape(rows,blocks,BLOCK)
    maximum=np.abs(grouped).max(axis=2)
    scales=maximum/np.float32(127)
    if np.any((maximum>0)&(scales==0)):
        raise ValueError('INT8 scale underflow')
    safe=np.where(scales==0,np.float32(1),scales)
    quantized=np.clip(np.rint(grouped/safe[:,:,None]),-127,127).astype(np.int8)
    return quantized.reshape(rows,blocks*BLOCK)[:,:cols].copy(),scales.astype('<f4',copy=False)


def quantize_file(source:Path,output:Path,chunk_rows=1024):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output or chunk_rows<=0:
        raise ValueError('use a distinct output and positive chunk size')
    if output.exists():raise FileExistsError(output)
    source_hash=sha256(source)
    output.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=output.name+'.',suffix='.tmp',dir=output.parent)
    os.close(descriptor)
    try:
        with safe_open(source,framework='pt',device='cpu') as reader:
            names=sorted(name for name in reader.keys() if MATRIX.fullmatch(name))
            if not names:raise ValueError('no supported decoder matrices')
            header={'__metadata__':{'format':FORMAT,'block_size':str(BLOCK),'source_sha256':source_hash}}
            offset=0
            for name in names:
                tensor=reader.get_slice(name)
                if tensor.get_dtype() not in ('F32','F16','BF16'):raise ValueError(f'unsupported source dtype: {name}')
                shape=tensor.get_shape()
                if len(shape)!=2 or 0 in shape:raise ValueError(f'invalid matrix {name}')
                rows,cols=shape;blocks=(cols+BLOCK-1)//BLOCK
                size=rows*cols
                header[name]={'dtype':'I8','shape':shape,'data_offsets':[offset,offset+size]};offset+=size
                size=rows*blocks*4
                header[name+SUFFIX]={'dtype':'F32','shape':[rows,blocks],'data_offsets':[offset,offset+size]};offset+=size
            raw=json.dumps(header,separators=(',',':')).encode()
            raw+=b' '*((-len(raw))%8)
            start=8+len(raw)
            with open(temporary,'wb') as stream:
                stream.write(struct.pack('<Q',len(raw)));stream.write(raw)
                stream.truncate(start+offset)
                for name in names:
                    rows,cols=header[name]['shape'];blocks=(cols+BLOCK-1)//BLOCK
                    tensor=reader.get_slice(name)
                    for row in range(0,rows,chunk_rows):
                        values=tensor[row:min(row+chunk_rows,rows),:].float().numpy()
                        q,s=quantize_blocks(values)
                        stream.seek(start+header[name]['data_offsets'][0]+row*cols)
                        stream.write(q.tobytes())
                        stream.seek(start+header[name+SUFFIX]['data_offsets'][0]+row*blocks*4)
                        stream.write(s.tobytes())
                stream.flush();os.fsync(stream.fileno())
        # Re-open with the upstream safetensors validator before publication.
        with safe_open(temporary,framework='np') as verified:
            if len(verified.keys())!=len(names)*2:raise ValueError('incomplete export')
        if sha256(source)!=source_hash:raise ValueError('source changed during export')
        result={'schema_version':1,'format':FORMAT,'block_size':BLOCK,'source_sha256':source_hash,
                'output_sha256':sha256(Path(temporary)),'matrix_count':len(names),'bytes':Path(temporary).stat().st_size}
        os.link(temporary,output)
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/upstream/model-00001-of-000001.safetensors'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=quantize_file(args.source,args.output)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
