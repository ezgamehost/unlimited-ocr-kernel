"""Prepare bounded-memory, versioned quantized decoder overlays; leave source intact."""
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


def quantize_blocks(values, bits=8):
    if type(bits) is not int or bits not in (4,8):
        raise ValueError('weight quantization bits must be 4 or 8')
    bound=(1<<(bits-1))-1
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('quantization needs a finite nonempty matrix')
    rows,cols=x.shape
    blocks=(cols+BLOCK-1)//BLOCK
    padded=np.zeros((rows,blocks*BLOCK),dtype=np.float32)
    padded[:,:cols]=x
    grouped=padded.reshape(rows,blocks,BLOCK)
    maximum=np.abs(grouped).max(axis=2)
    scales=maximum/np.float32(bound)
    if np.any((maximum>0)&(scales==0)):
        raise ValueError('INT8 scale underflow')
    safe=np.where(scales==0,np.float32(1),scales)
    quantized=np.clip(np.rint(grouped/safe[:,:,None]),-bound,bound).astype(np.int8)
    return quantized.reshape(rows,blocks*BLOCK)[:,:cols].copy(),scales.astype('<f4',copy=False)


def quantize_file(source:Path,output:Path,chunk_rows=1024,*,bits=8,packed=False,q8_head=False):
    if type(bits) is not int or bits not in (4,8):
        raise ValueError('weight quantization bits must be 4 or 8')
    if type(packed) is not bool or (packed and bits!=4):
        raise ValueError('packing requires 4-bit weights')
    if type(q8_head) is not bool or (q8_head and not (packed and bits==4)):
        raise ValueError('Q8 head requires packed 4-bit body')
    format_name='uocr-q4-q8head-block32-v1' if q8_head else ('uocr-q4-block32-v1' if packed else FORMAT)
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
            if q8_head and 'lm_head.weight' not in names:raise ValueError('hybrid requires lm_head.weight')
            header={'__metadata__':{'format':format_name,'block_size':str(BLOCK),'source_sha256':source_hash}}
            if q8_head:
                header['__metadata__'].update(body_bits='4',head_bits='8')
            elif packed:
                header['__metadata__'].update(weight_bits='4',storage_bits='4')
            elif bits==4:
                header['__metadata__'].update(simulated_weight_bits='4',storage_bits='8')
            offset=0
            for name in names:
                matrix_packed=packed and not (q8_head and name=='lm_head.weight')
                suffix='.q4_scales' if matrix_packed else SUFFIX
                tensor=reader.get_slice(name)
                if tensor.get_dtype() not in ('F32','F16','BF16'):raise ValueError(f'unsupported source dtype: {name}')
                shape=tensor.get_shape()
                if len(shape)!=2 or 0 in shape:raise ValueError(f'invalid matrix {name}')
                rows,cols=shape;blocks=(cols+BLOCK-1)//BLOCK
                size=rows*blocks*16 if matrix_packed else rows*cols
                header[name]={'dtype':'U8' if matrix_packed else 'I8',
                              'shape':[rows,blocks,16] if matrix_packed else shape,
                              'data_offsets':[offset,offset+size]};offset+=size
                size=rows*blocks*4
                header[name+suffix]={'dtype':'F32','shape':[rows,blocks],'data_offsets':[offset,offset+size]};offset+=size
            raw=json.dumps(header,separators=(',',':')).encode()
            raw+=b' '*((-len(raw))%8)
            start=8+len(raw)
            with open(temporary,'wb') as stream:
                stream.write(struct.pack('<Q',len(raw)));stream.write(raw)
                stream.truncate(start+offset)
                for name in names:
                    matrix_packed=packed and not (q8_head and name=='lm_head.weight')
                    matrix_bits=8 if q8_head and name=='lm_head.weight' else bits
                    suffix='.q4_scales' if matrix_packed else SUFFIX
                    tensor=reader.get_slice(name)
                    rows,cols=tensor.get_shape();blocks=(cols+BLOCK-1)//BLOCK
                    for row in range(0,rows,chunk_rows):
                        values=tensor[row:min(row+chunk_rows,rows),:].float().numpy()
                        q,s=quantize_blocks(values,matrix_bits)
                        if matrix_packed:
                            padded=np.zeros((q.shape[0],blocks*BLOCK),dtype=np.int8)
                            padded[:,:cols]=q
                            codes=padded.reshape(q.shape[0],blocks,BLOCK).view(np.uint8)
                            q=(codes[:,:,:16]&15)|((codes[:,:,16:]&15)<<4)
                        stride=blocks*16 if matrix_packed else cols
                        stream.seek(start+header[name]['data_offsets'][0]+row*stride)
                        stream.write(q.tobytes())
                        stream.seek(start+header[name+suffix]['data_offsets'][0]+row*blocks*4)
                        stream.write(s.tobytes())
                stream.flush();os.fsync(stream.fileno())
        # Re-open with the upstream safetensors validator before publication.
        with safe_open(temporary,framework='np') as verified:
            if len(verified.keys())!=len(names)*2:raise ValueError('incomplete export')
        if sha256(source)!=source_hash:raise ValueError('source changed during export')
        result={'schema_version':1,'format':format_name,'block_size':BLOCK,'source_sha256':source_hash,
                'output_sha256':sha256(Path(temporary)),'matrix_count':len(names),'bytes':Path(temporary).stat().st_size}
        if q8_head:
            result.update(body_bits=4,head_bits=8)
        elif packed:
            result.update(weight_bits=4,storage_bits=4)
        elif bits==4:
            result.update(simulated_weight_bits=4,storage_bits=8,
                          purpose='INT4 accuracy simulation in INT8 storage; no packing or speedup claim')
        os.link(temporary,output)
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/upstream/model-00001-of-000001.safetensors'))
    parser.add_argument('--output',type=Path,required=True)
    precision=parser.add_mutually_exclusive_group()
    precision.add_argument('--simulate-int4',action='store_true',
                        help='Use signed 4-bit weight levels in unchanged INT8 storage; accuracy experiment only')
    precision.add_argument('--int4',action='store_true',help='Write packed block32 signed INT4 weights')
    precision.add_argument('--int4-q8-head',action='store_true',help='Packed INT4 body with Q8 output head')
    args=parser.parse_args()
    result=quantize_file(args.source,args.output,bits=4 if args.simulate_int4 or args.int4 or args.int4_q8_head else 8,packed=args.int4 or args.int4_q8_head,q8_head=args.int4_q8_head)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
