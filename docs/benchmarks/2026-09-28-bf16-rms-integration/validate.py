"""Sequential correctness gates; run only after performance children finish."""
import json
import os
from pathlib import Path
import subprocess
from tools.benchmark import _stop_process_group
from tools.benchmark_resident import wait_child
from tools.fetch_model import sha256

root = Path(__file__).resolve().parent
output = root / 'validation'
output.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(RUSTUP_HOME=str(Path('.tools/rustup').resolve()), CARGO_HOME=str(Path('.tools/cargo').resolve()), VECLIB_MAXIMUM_THREADS='8', OMP_NUM_THREADS='8')
cargo = str(Path('.tools/cargo/bin/cargo').absolute())
steps = [
    ('fmt', [cargo, 'fmt', '--check']),
    ('rust-tests', [cargo, 'test', '--release']),
    ('clippy', [cargo, 'clippy', '--all-targets', '--', '-D', 'warnings']),
    ('expert-reference', [cargo, 'test', '--release', '--test', 'decoder', 'actual_expert_scheduling', '--', '--ignored', '--nocapture', '--test-threads=1']),
    ('decoder-reference', [cargo, 'test', '--release', '--test', 'decoder', 'actual_decoder_layers', '--', '--ignored', '--nocapture', '--test-threads=1']),
    ('teacher-logits', [cargo, 'test', '--release', '--test', 'decoder', 'actual_decoder_teacher_forced_logits_match_upstream', '--', '--ignored', '--exact', '--nocapture']),
    ('expert-routes', [cargo, 'test', '--release', '--lib', 'actual_', '--', '--ignored', '--nocapture', '--test-threads=1']),
    ('q8-digest', [cargo, 'test', '--release', '--lib', 'model::benchmark::fixed_work_decoder_benchmark', '--', '--ignored', '--exact', '--nocapture']),
    ('build', [cargo, 'build', '--release', '--bin', 'uocr']),
]
records = []
for name, command in steps:
    with (output / (name + '.log')).open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            resources = wait_child(child, 1200)
        except BaseException:
            _stop_process_group(child)
            raise
    records.append(dict(name=name, command=command, resources=resources))
    (output / 'steps.json').write_text(json.dumps(records, indent=2) + '\n')
    assert resources['exit_code']==0 and not resources['forced_kill'], name
    print(name + ' passed', flush=True)
(output / 'identity.json').write_text(json.dumps(dict(cli_sha256=sha256(Path('target/release/uocr')), source_sha256={str(p):sha256(p) for p in sorted(Path('src').rglob('*.rs'))}), indent=2) + '\n')
