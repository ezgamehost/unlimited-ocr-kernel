"""Download the pinned original model, publishing files only after verification."""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
REPOSITORY = "baidu/Unlimited-OCR"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256=None) -> str:
    if destination.exists() and expected_sha256 and sha256(destination) == expected_sha256:
        return expected_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        os.replace(temporary, destination)
        return actual
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/upstream"))
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    api = f"https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}"
    with urllib.request.urlopen(api, timeout=30) as response:
        files = json.load(response)
    manifest = {"schema_version": 1, "repository": REPOSITORY, "revision": REVISION, "files": {}}
    for entry in files:
        name = entry["path"]
        if entry["type"] != "file" or "/" in name or not (name.endswith((".json", ".py", ".safetensors")) or name == "LICENSE"):
            continue
        if args.skip_weights and name.endswith(".safetensors"):
            continue
        print(f"fetch {name} ({entry['size']} bytes)", flush=True)
        digest = download(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                          args.output / name, entry.get("lfs", {}).get("oid"))
        manifest["files"][name] = {"sha256": digest, "size": entry["size"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
