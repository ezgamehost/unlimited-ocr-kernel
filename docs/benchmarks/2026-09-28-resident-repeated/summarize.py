"""Reproduce the completed resident comparison from adjacent raw records."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    values = sorted(values)
    return {"median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1]}


def engine(name):
    directory = ROOT / name
    summary, process = read(directory / "summary.json"), read(directory / "process.json")
    assert summary["complete"] and summary["protocol_complete"]
    assert summary["expected_requests"] == summary["collected_requests"] == 48
    assert not process["resources"]["forced_kill"]
    assert process["resources"]["peak_rss_bytes"] > 0
    assert process["resources"]["exit_code"] in ([0] if name == "native" else [0, -15])
    rows = [read(p) for p in sorted(directory.glob("*.record.json"))]
    assert len(rows) == 48
    assert all(r["status"] == "ok" and r["finish_reason"] == "eos" for r in rows)
    if name == "llama":
        assert all(r["cache_tokens"] == 0 and r["timings"]["prompt_n"] == 1518 for r in rows)
    result = {"resources": process["resources"],
              "startup_wall_seconds": process["startup_wall_seconds"], "images": {}}
    outputs = {}
    for image in ("english", "chinese", "table", "dense"):
        selected = [r for r in rows if r["image_id"] == image and not r["warmup"]]
        assert sorted(r["iteration"] for r in selected) == list(range(10))
        decoded = [read(directory / f"{r['index']:03d}-{image}.decoded.json") for r in selected]
        assert all(d == decoded[0] for d in decoded)
        outputs[image] = decoded[0]
        result["images"][image] = {
            "cer": sorted({r["cer"] for r in selected}),
            "emitted_tokens": sorted({r["emitted_tokens"] for r in selected}),
            "wall_seconds": distribution(r["wall_seconds"] for r in selected),
            "first_token_seconds": distribution(r["first_token_seconds"] for r in selected),
            "decode_tokens_per_second": distribution(r["decode_steps"] / r["decode_seconds"] for r in selected),
        }
    return result, outputs, process


def main():
    native, native_outputs, np = engine("native")
    llama, llama_outputs, lp = engine("llama")
    assert np["schedule"] == lp["schedule"]
    assert np["thread_environment"] == lp["thread_environment"]
    assert np["manifest"] == lp["manifest"]
    comparison = {}
    for image in native_outputs:
        n, l = native["images"][image], llama["images"][image]
        assert n["cer"] == l["cer"] and n["emitted_tokens"] == l["emitted_tokens"]
        nt, lt = native_outputs[image]["token_ids"], llama_outputs[image]["token_ids"]
        comparison[image] = {
            "median_speed_ratio_llama_over_native": l["wall_seconds"]["median"] / n["wall_seconds"]["median"],
            "token_id_differences": sum(a != b for a, b in zip(nt, lt)) + abs(len(nt) - len(lt)),
            "exact_raw_text_match": native_outputs[image]["text"] == llama_outputs[image]["text"],
        }
    output = {"native": native, "llama": llama, "comparison": comparison,
              "notes": ["10 measured requests per image; 2 warmups excluded.",
                        "Nearest-rank p95 is the maximum with ten samples.",
                        "Native TTFT uses its internal clock; llama uses first streamed token arrival.",
                        "Native prefix-plus-128 ring differs from llama full attention.",
                        "Native block32 INT8/dynamic activations/BF16 vision differs from Q8_0/F16 projector.",
                        "Sequential engine runs on a shared Mac; no host-isolation claim."]}
    (ROOT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
