"""Reproducible, process-per-request CPU OCR benchmarking."""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

from tools.score import character_error_rate, normalize_ocr


def benchmark_environment(threads: int, blas_threads: int) -> dict:
    """Configure only the child, before Accelerate can cache its thread policy."""
    if threads < 1 or blas_threads < 1 or blas_threads > threads:
        raise ValueError("require 1 <= BLAS threads <= worker threads")
    environment = os.environ.copy()
    environment["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    environment["OMP_NUM_THREADS"] = str(threads)
    return environment


def completion_text_without_eos(response: dict, eos_piece: str, eos_id: int = 1) -> tuple:
    """Adapt --special native server output; preserve raw response and literal text."""
    text, tokens = response["content"], response["tokens"]
    if response["stop_type"] == "eos" and tokens and tokens[-1] == eos_id:
        if not eos_piece or not text.endswith(eos_piece):
            raise ValueError("EOS token and rendered terminal marker disagree")
        return text[:-len(eos_piece)], tokens[:-1]
    return text, list(tokens)


def validate_image(path: Path, expected_sha256: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"image hash mismatch: {path}")


def aggregate(records: list) -> dict:
    durations = []
    failed = truncated = 0
    for record in records:
        if record.get("status") != "ok":
            failed += 1
        elif record.get("finish_reason") != "eos":
            truncated += 1
        else:
            value = record["wall_seconds"]
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid benchmark duration")
            durations.append(value)
    durations.sort()
    return {
        "completed": len(durations), "failed": failed, "truncated": truncated,
        "median_seconds": statistics.median(durations) if durations else None,
        "p95_seconds": durations[math.ceil(0.95 * len(durations)) - 1] if durations else None,
    }


def run_llama(command: list, output_prefix: Path, token_limit: int, timeout: float,
              *, environment: dict | None = None) -> dict:
    """Measure a fresh process; preserve partial logs on failures/timeouts."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with output_prefix.with_suffix(".stdout.txt").open("wb") as out, \
            output_prefix.with_suffix(".stderr.txt").open("wb") as err:
        try:
            options = {} if environment is None else {"env": environment}
            process = subprocess.run(["/usr/bin/time", "-l"] + command,
                                     stdout=out, stderr=err, timeout=timeout, **options)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    elapsed = time.monotonic() - started
    log = output_prefix.with_suffix(".stderr.txt").read_text(errors="replace")
    output = output_prefix.with_suffix(".stdout.txt").read_text(errors="replace").strip()
    steps = re.search(r"\beval time\s*=\s*([\d.]+) ms /\s*(\d+) runs", log)
    rss = re.search(r"(\d+)\s+maximum resident set size", log)
    vision = re.findall(r"mtmd batch encoding done in (\d+) ms", log)
    count = int(steps[2]) if steps else None
    # The first token can come from prefill, leaving cap-1 eval runs when the
    # output cap is reached. Timing alone cannot distinguish that boundary
    # from EOS: exclude it from completed latency aggregates conservatively.
    finish = "error" if code != 0 else (
        "unknown" if count is None or count == token_limit - 1
        else "token_limit" if count >= token_limit else "eos")
    return {
        "status": "ok" if code == 0 else "error", "exit_code": code,
        "timed_out": timed_out, "finish_reason": finish,
        "wall_seconds": elapsed, "decode_steps": count,
        "decode_seconds": float(steps[1]) / 1000 if steps else None,
        "vision_seconds": sum(map(int, vision)) / 1000 if vision else None,
        "peak_rss_bytes": int(rss[1]) if rss else None,
        "output": output, "command": command,
        "thread_environment": {key: (os.environ if environment is None else environment).get(key)
                               for key in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--blas-threads", type=int, default=1,
                        help="child Accelerate thread limit; default 1 prevents nested BLAS workers")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.threads < 1 or args.repetitions < 1 or args.warmups < 0 or args.max_tokens < 1:
        parser.error("threads, repetitions and max-tokens must be positive; warmups nonnegative")
    try:
        environment = benchmark_environment(args.threads, args.blas_threads)
    except ValueError as error:
        parser.error(str(error))
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for item in manifest["images"]:
        image = (args.manifest.parent / item["path"]).resolve()
        validate_image(image, item["sha256"])
        for iteration in range(-args.warmups, args.repetitions):
            label = f"{item['id']}-{iteration + args.warmups:03}"
            command = [str(args.llama.resolve()), "-m", str(args.model.resolve()),
                       "--mmproj", str(args.mmproj.resolve()), "--image", str(image),
                       "-p", "document parsing.", "-ngl", "0", "--device", "none",
                       "--no-mmproj-offload", "--no-op-offload", "--no-kv-offload",
                       "-t", str(args.threads), "-tb", str(args.threads),
                       "-c", "32768", "-n", str(args.max_tokens), "--temp", "0",
                       "--seed", "42", "--repeat-penalty", "1.0", "--no-warmup", "--jinja", "-lv", "4"]
            record = run_llama(command, args.output / label, args.max_tokens, args.timeout,
                               environment=environment)
            record.update(image_id=item["id"], iteration=iteration, warmup=iteration < 0,
                          image_sha256=item["sha256"], process_mode="fresh-process-warm-files")
            record["cer"] = character_error_rate(normalize_ocr(item["text"]), normalize_ocr(record["output"]))
            (args.output / f"{label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: record[k] for k in ["image_id", "iteration", "status", "finish_reason", "wall_seconds", "cer"]}), flush=True)
            if iteration >= 0:
                records.append(record)
    summary = {"schema_version": 1, "system": platform.platform(),
               "aggregate": aggregate(records),
               "per_image": {item["id"]: aggregate([r for r in records if r["image_id"] == item["id"]])
                             for item in manifest["images"]},
               "notes": ["full-attention llama.cpp; n-gram suppression disabled", "fresh processes, filesystem cache not cleared"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
