import base64,json,os,signal,socket,subprocess,time,urllib.request,urllib.error
from pathlib import Path
from tools.benchmark import benchmark_environment, completion_text_without_eos
from tools.score import character_error_rate,normalize_ocr
import argparse
parser = argparse.ArgumentParser(description="Reproduce the five-request resident pilot")
parser.add_argument("--output", type=Path, required=True)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=False)
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
cmd=['/tmp/unlimited-ocr-baseline/bin/llama-b11223/llama-server','-m','/tmp/unlimited-ocr-baseline/models/model.gguf','--mmproj','artifacts/mmproj-unlimited-ocr-F16-tiles32.gguf','-ngl','0','--device','none','--no-mmproj-offload','--no-op-offload','--no-kv-offload','-t','8','-tb','8','-c','32768','-n','8192','--temp','0','--seed','42','--repeat-penalty','1','--no-warmup','--special','--jinja','-lv','4','--host','127.0.0.1','--port',str(port),'--parallel','1','--no-cache-prompt','--cache-ram','0','--no-cache-idle-slots','--no-context-shift','--offline']
env=benchmark_environment(8,8);records=[];stamp=time.monotonic()
with (root/'server.stdout').open('w') as out,(root/'server.stderr').open('w') as err:
 p=subprocess.Popen(['/usr/bin/time','-l']+cmd,env=env,stdout=out,stderr=err,start_new_session=True)
 try:
  while True:
   if p.poll() is not None:raise RuntimeError('server exited before ready')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
     if response.status==200:break
   except (urllib.error.URLError,TimeoutError):pass
   if time.monotonic()-stamp>120:raise TimeoutError('server readiness')
   time.sleep(.1)
  startup=time.monotonic()-stamp;print('ready',startup,flush=True)
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/props',timeout=5) as response:props=json.load(response)
  (root/'props.json').write_text(json.dumps(props,indent=2)+'\n')
  marker=props['media_marker'];assert props['modalities']['vision']
  for index,name in enumerate(['english','english','chinese','table','dense']):
   encoded=base64.b64encode(Path(f'benchmarks/corpus/{name}.png').read_bytes()).decode()
   payload={'prompt':{'prompt_string':marker+'document parsing.','multimodal_data':[encoded]},'temperature':0,'seed':42,'repeat_penalty':1,'n_predict':8192,'cache_prompt':False,'return_tokens':True,'stream':False}
   request=urllib.request.Request(f'http://127.0.0.1:{port}/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
   started=time.monotonic()
   try:
    with urllib.request.urlopen(request,timeout=180) as response:raw=response.read()
   except urllib.error.HTTPError as error:
    (root/f'{index}-{name}-error.json').write_bytes(error.read());raise
   wall=time.monotonic()-started;(root/f'{index}-{name}.json').write_bytes(raw)
   result=json.loads(raw);text,tokens=completion_text_without_eos(result,props['eos_token']);timings=result.get('timings',{})
   (root/f'{index}-{name}-decoded.json').write_text(json.dumps(dict(text=text,token_ids=tokens),ensure_ascii=False,indent=2)+'\n')
   record=dict(image=name,index=index,request_wall_seconds=wall,finish=result['stop_type'],tokens=len(result.get('tokens',[])),emitted_tokens_excluding_eos=len(tokens),tokens_evaluated=result.get('tokens_evaluated'),timings=timings,cer=character_error_rate(normalize_ocr(Path(f'benchmarks/corpus/{name}.txt').read_text()),normalize_ocr(text)))
   records.append(record);(root/'summary.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
   assert timings.get('cache_n',0)==0
  (root/'process.json').write_text(json.dumps(dict(command=cmd,thread_environment={k:env[k] for k in ['VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']},startup_wall_seconds=startup,request_order=[r['image'] for r in records],process_mode='resident independent requests',max_tokens=8192),indent=2)+'\n')
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
