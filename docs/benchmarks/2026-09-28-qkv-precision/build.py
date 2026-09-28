"""Build identical fixed-work harnesses with sequential and concurrent QKV."""
import hashlib, json, os, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = Path.cwd()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
env = dict(os.environ, RUSTUP_HOME=str(REPO/'.tools/rustup'), CARGO_HOME=str(REPO/'.tools/cargo'))
cargo = str((REPO/'.tools/cargo/bin/cargo').absolute())
source = REPO/'src/decoder/mod.rs'
accepted = source.read_bytes()
cli_hash = sha(REPO/'target/release/uocr')
assert cli_hash == '1f1902e4db8a0c9ad5ad79242a439c589ba423ab8462cf52754e8af285c1f193'
items = {}
try:
    for variant in ['baseline', 'candidate']:
        source.write_bytes(subprocess.check_output(['git','show','21da757:src/decoder/mod.rs']) if variant == 'baseline' else accepted)
        with (ROOT/f'build-{variant}.log').open('w') as log:
            result = subprocess.run([cargo,'test','--release','--lib','--no-run','--message-format=json'],env=env,stdout=subprocess.PIPE,stderr=log,text=True)
        (ROOT/f'build-{variant}.jsonl').write_text(result.stdout)
        result.check_returncode()
        artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        paths = [a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a.get('profile',{}).get('test')]
        assert len(paths)==1, paths
        target = REPO/'artifacts/qkv-precision'/variant
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(paths[0],target)
        items[variant] = dict(binary=str(target),sha256=sha(target),decoder_source_sha256=sha(source))
finally:
    source.write_bytes(accepted)
assert sha(REPO/'target/release/uocr')==cli_hash
(ROOT/'identity.json').write_text(json.dumps(dict(binaries=items,harness_sha256=sha(REPO/'src/model/benchmark.rs'),cli_sha256=cli_hash),indent=2)+'\n')
(ROOT/'harness.diff').write_bytes(subprocess.check_output(['git','diff','--','src/model/benchmark.rs']))
print(json.dumps(items),flush=True)
