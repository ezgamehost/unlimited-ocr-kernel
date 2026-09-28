"""Fresh full-page ABBA using the accepted scheduling runtime."""
import contextlib
import datetime
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
SERIES=ROOT/'powered-abba-1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def snapshot():
    commands=[['pmset','-g','batt'],['pmset','-g','custom'],['pmset','-g','therm'],['sysctl','vm.swapusage'],['vm_stat']]
    values={}
    for command in commands:
        result=subprocess.run(command, capture_output=True, text=True, check=False)
        values[' '.join(command)]=dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), commands=values)


def require_power(state):
    battery=state['commands']['pmset -g batt']
    assert battery['returncode']==0 and "Now drawing from 'AC Power'" in battery['stdout']
    profile=state['commands']['pmset -g custom']
    modes=re.findall(r'^\s*powermode\s+(\d+)\s*$',profile['stdout'],re.MULTILINE)
    assert profile['returncode']==0 and modes and all(mode=='0' for mode in modes)


def invoke(path, args):
    spec=importlib.util.spec_from_file_location('benchmark_stage', path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv=[str(path),*args]
    module.main()


assert not SERIES.exists() and not (ROOT/'host-before.json').exists(), 'Use a fresh namespace; retain existing evidence'
before=snapshot()
require_power(before)
save(ROOT/'host-before.json',before)
status=dict(state='running', source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
save(ROOT/'status.json',status)
try:
    invoke(ROOT.parent/'2026-09-28-packed-int4-resident/run_repeated.py', [
        '--native-precision','int8',
        '--quality-proof',str(ROOT.parent/'2026-09-28-concurrent-qkv/quality/q8'),
        '--output-root',str(SERIES)])
except BaseException as error:
    status.update(state='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    after=snapshot()
    save(ROOT/'host-after.json',after)
    save(ROOT/'status.json',status)
require_power(after)
with (ROOT/'audit.log').open('w') as log, contextlib.redirect_stdout(log):
    invoke(ROOT.parent/'2026-09-28-packed-int4-resident/summarize_repeated.py',['--input-root',str(SERIES)])
status.update(state='complete', audited_requests=96)
save(ROOT/'status.json',status)
print(json.dumps(status),flush=True)
