import json,subprocess,time
from tools.benchmark import benchmark_environment
from pathlib import Path
from tools.score import character_error_rate,normalize_ocr
import argparse
parser = argparse.ArgumentParser(description="Reproduce the five-request resident pilot")
parser.add_argument("--output", type=Path, required=True)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=False)
cmd=['/usr/bin/time','-l','target/release/uocr','--jsonl','--threads','8','--int8-weights','artifacts/unlimited-ocr-q8-block32.safetensors','--int8-activations']
start=time.monotonic();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=benchmark_environment(8,8))
rows=[]
try:
 initial=p.stderr.readline()
 event=json.loads(initial);assert event['event']=='loaded'
 startup=time.monotonic()-start
 for index,name in enumerate(['english','english','chinese','table','dense']):
  request={'image':f'benchmarks/corpus/{name}.png','prompt':'<image>\ndocument parsing.','generation':{'max_tokens':8192}}
  stamp=time.monotonic();p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
  line=p.stdout.readline();wall=time.monotonic()-stamp
  (root/f'{index}-{name}.json').write_text(line)
  response=json.loads(line);assert response['status']=='ok',response
  result=response['result'];old=json.load(open(f'docs/benchmarks/2026-09-28-shared-expert/fresh/{name}.json'))
  row=dict(case=name,index=index,request=request,request_wall_seconds=wall,timings=result['timings'],finish=result['finish_reason'],tokens=len(result['token_ids']),same_ids_as_old_unmatched_prompt=result['token_ids']==old['token_ids'],cer=character_error_rate(normalize_ocr(Path(f'benchmarks/corpus/{name}.txt').read_text()),normalize_ocr(result['text'])))
  rows.append(row);(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 p.stdin.close();rest=p.stderr.read();code=p.wait();(root/'process.stderr').write_text(initial+rest)
 (root/'process.json').write_text(json.dumps(dict(command=cmd,thread_environment={'VECLIB_MAXIMUM_THREADS':'8','OMP_NUM_THREADS':'8'},mode='resident independent requests',load_event=event,startup_wall_seconds=startup,process_wall_seconds=time.monotonic()-start,exit_code=code),indent=2)+'\n')
 assert code==0
 assert all(r['finish']=='eos' for r in rows)
finally:
 if p.poll() is None:p.kill();p.wait()
