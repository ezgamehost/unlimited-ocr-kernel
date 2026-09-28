"""Copy a llama.cpp projector with an explicit crop cap; verify unchanged tensors."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails,copy_with_new_metadata
from tools.fetch_model import sha256

KEY='clip.vision.preproc_max_tiles'


def copy_with_crop_limit(source: Path,target: Path,max_tiles: int) -> dict:
    source,target=Path(source).resolve(),Path(target).resolve()
    if source==target or not 2<=max_tiles<=256:
        raise ValueError('use a distinct output and a crop cap between 2 and 256')
    if target.exists():
        raise FileExistsError(target)
    reader=gguf.GGUFReader(source,'r')
    if reader.get_field('general.architecture').contents()!='clip':
        raise ValueError('input is not a CLIP projector')
    source_hash=sha256(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=target.parent)
    os.close(descriptor)
    try:
        writer=gguf.GGUFWriter(temporary,arch='clip',endianess=reader.endianess)
        copy_with_new_metadata(reader,writer,{KEY:MetadataDetails(gguf.GGUFValueType.UINT32,max_tiles)},[])
        copied=gguf.GGUFReader(temporary,'r')
        if len(reader.tensors)!=len(copied.tensors):
            raise ValueError('tensor count changed')
        hashes=[]
        for before,after in zip(reader.tensors,copied.tensors):
            if before.name!=after.name or before.tensor_type!=after.tensor_type or list(before.shape)!=list(after.shape):
                raise ValueError('tensor descriptor changed')
            old=hashlib.sha256(memoryview(before.data).cast('B')).hexdigest()
            new=hashlib.sha256(memoryview(after.data).cast('B')).hexdigest()
            if old!=new:
                raise ValueError(f'tensor bytes changed: {before.name}')
            hashes.append({'name':before.name,'sha256':old})
        names=lambda r:{k for k in r.fields if not k.startswith('GGUF.') and k!=KEY}
        if names(reader)!=names(copied):
            raise ValueError('unrelated metadata keys changed')
        for name in names(reader):
            a,b=reader.get_field(name),copied.get_field(name)
            if a.types!=b.types or a.contents()!=b.contents():
                raise ValueError(f'metadata changed: {name}')
        if copied.get_field(KEY).contents()!=max_tiles:
            raise ValueError('crop cap not written')
        if sha256(source)!=source_hash:
            raise ValueError('source changed during copy')
        result={'schema_version':1,'source':str(source),'source_sha256':source_hash,
                'output':str(target),'output_sha256':sha256(Path(temporary)),
                'max_tiles':max_tiles,'tensor_count':len(hashes),'tensors':hashes}
        # Hard-link publication fails atomically if another writer created target.
        os.link(temporary,target)
        return result
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--max-tiles',type=int,default=32)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    result=copy_with_crop_limit(args.source,args.output,args.max_tiles)
    args.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}))


if __name__=='__main__':main()
