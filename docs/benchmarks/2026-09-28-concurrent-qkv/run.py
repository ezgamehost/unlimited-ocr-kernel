"""Fixed-work Q8 decoder ABBA for the concurrent QKV projection scheduling."""
import argparse
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

from tools.benchmark import benchmark_environment, _stop_process_group
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def power(source):
    battery = subprocess.check_output(['pmset','-g','batt'],text=True)
    expected = 'AC Power' if source=='ac' else 'Battery Power'
    assert f"Now drawing from '{expected}'" in battery,battery
    profile = subprocess.check_output(['pmset','-g','custom'],text=True)
    modes = re.findall(r'^\s*powermode\s+(\d+)\s*$',profile,re.MULTILINE)
    assert modes and all(mode=='0' for mode in modes)
    if source=='battery':
        charge = re.search(r'\t(\d+)%;',battery)
        assert charge and int(charge[1])>=50
    return dict(battery=battery,profile=profile)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power-source',choices=['ac','battery'],default='ac')
    parser.add_argument('--output-name',default='powered-abba-1')
    args=parser.parse_args()
    power(args.power_source)
    output=ROOT/args.output_name
    output.mkdir(exist_ok=False)
    baseline=json.loads((ROOT/'baseline.json').read_text())
    log=(ROOT/'focused-tests.log').read_text()
    executable=Path(re.search(r'Running unittests src/lib.rs \(([^)]+)\)',log)[1])
    candidate=Path('artifacts/concurrent-qkv/candidate')
    if not candidate.exists():
        shutil.copy2(executable,candidate)
    else:
        assert sha256(candidate)==sha256(executable)
    binaries=dict(baseline=baseline,candidate=dict(binary=str(candidate.resolve()),sha256=sha256(candidate)))
    for item in binaries.values():
        assert sha256(Path(item['binary']))==item['sha256']
    overlay=Path('artifacts/unlimited-ocr-q8-block32.safetensors')
    assert sha256(overlay)=='b6afd272e6f19de970f97245c83b09e110a20acc8033f2853d0e6739e5fde441'
    save(output/'identity.json',dict(binaries=binaries,overlay_sha256=sha256(overlay),
         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         candidate_source_sha256={str(p):sha256(p) for p in sorted(Path('src').rglob('*.rs'))},
         extra_model_storage_bytes=0,power_source=args.power_source))
    (output/'candidate.diff').write_bytes(subprocess.check_output(['git','diff','--','src','tests/quantized.rs']))
    reports=[]
    for phase,name in enumerate(['baseline','candidate','candidate','baseline']):
        before=power(args.power_source)
        binary=binaries[name]
        assert sha256(Path(binary['binary']))==binary['sha256']
        command=[binary['binary'],'model::benchmark::fixed_work_decoder_benchmark','--ignored','--exact','--nocapture','--test-threads=1']
        env=benchmark_environment(8,8)
        env.update(UOCR_BENCH_WEIGHTS=str(overlay),UOCR_BENCH_THREADS='8')
        path=output/f'{phase}-{name}'
        started=time.monotonic()
        with path.with_suffix('.log').open('wb') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            try:
                resources=wait_child(child,900)
            except BaseException:
                _stop_process_group(child)
                raise
        wall=time.monotonic()-started
        row=dict(phase=phase,variant=name,command=command,resources=resources,power_before=before,
                 process_wall_seconds=wall)
        save(path.with_suffix('.json'),row)
        row['power_after']=power(args.power_source)
        row['samples']=[json.loads(line[line.index('{'):]) for line in path.with_suffix('.log').read_text().splitlines()
                        if '"benchmark":"native_fixed_work_decoder"' in line]
        save(path.with_suffix('.json'),row)
        assert resources['exit_code']==0 and not resources['forced_kill']
        assert len(row['samples'])==4
        for i,sample in enumerate(row['samples']):
            assert sample['repetition']==i and sample['warmup']==(i==0)
            assert sample['decode_steps']==512 and sample['prefix_tokens']==1517 and sample['threads']==8
            assert sample['precision']=='int8_weights_dynamic_activations'
        reports.append(row)
        print(json.dumps(dict(phase=phase,variant=name,decode_seconds=[s['decode_with_checksum_seconds'] for s in row['samples']])),flush=True)
    hashes={s['logits_sha256'] for row in reports for s in row['samples']}
    assert len(hashes)==1,'candidate changed fixed-work logits'
    summary=dict(power_source=args.power_source,all_16_logit_hashes_equal=True,logits_sha256=next(iter(hashes)),variants={})
    for name in binaries:
        samples=[s for row in reports if row['variant']==name for s in row['samples'] if not s['warmup']]
        summary['variants'][name]={key:dict(median=statistics.median(s[key] for s in samples),
                                          minimum=min(s[key] for s in samples),maximum=max(s[key] for s in samples),n=len(samples))
                                  for key in ['decode_with_checksum_seconds','prefill_seconds']}
    summary['notes']=['Includes all-logit hashing in decode.', 'Process lifetime includes load/prepare, fixture setup and four timed repetitions; not an isolated startup measurement.',
                      'Battery results are exploratory; AC confirmation and full OCR gates required before speed claims.']
    save(output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
