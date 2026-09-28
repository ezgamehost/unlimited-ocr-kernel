import json,subprocess,time
from pathlib import Path
from tools.score import character_error_rate,normalize_ocr
root=Path('docs/benchmarks/2026-09-28-attention-heads/guarded-resident');root.mkdir(exist_ok=True)
cmd=['/usr/bin/time','-l','target/release/uocr','--jsonl','--threads','8','--int8-weights','artifacts/unlimited-ocr-q8-block32.safetensors','--int8-activations']
start=time.monotonic();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
rows=[]
try:
 initial=p.stderr.readline()
 event=json.loads(initial);assert event['event']=='loaded'
 startup=time.monotonic()-start
 for name in ['english','chinese','table','dense']:
  request={'image':f'benchmarks/corpus/{name}.png','generation':{'max_tokens':8192}}
  stamp=time.monotonic();p.stdin.write(json.dumps(request)+'\n');p.stdin.flush()
  line=p.stdout.readline();wall=time.monotonic()-stamp
  (root/f'{name}.json').write_text(line)
  response=json.loads(line);assert response['status']=='ok',response
  result=response['result'];old=json.load(open(f'docs/benchmarks/2026-09-28-shared-expert/fresh/{name}.json'))
  row=dict(case=name,request=request,request_wall_seconds=wall,timings=result['timings'],finish=result['finish_reason'],tokens=len(result['token_ids']),same_token_ids=result['token_ids']==old['token_ids'],cer=character_error_rate(normalize_ocr(Path(f'benchmarks/corpus/{name}.txt').read_text()),normalize_ocr(result['text'])))
  rows.append(row);(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 p.stdin.close();rest=p.stderr.read();code=p.wait();(root/'process.stderr').write_text(initial+rest)
 (root/'process.json').write_text(json.dumps(dict(command=cmd,mode='resident independent requests',load_event=event,startup_wall_seconds=startup,process_wall_seconds=time.monotonic()-start,exit_code=code),indent=2)+'\n')
 assert code==0
 assert all(r['same_token_ids'] and r['finish']=='eos' for r in rows)
finally:
 if p.poll() is None:p.kill();p.wait()
