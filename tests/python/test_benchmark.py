"""Catch biased aggregation, corrupt fixture acceptance and inaccurate scoring."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.benchmark import aggregate, validate_image, run_llama, benchmark_environment, completion_text_without_eos
from tools.score import character_error_rate, normalize_ocr


class BenchmarkTests(unittest.TestCase):
    def test_timeout_stops_child_and_sigterm_ignoring_grandchild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grandchild = root / 'grandchild.py'
            grandchild.write_text(
                "import os,signal,sys,time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                "print('grandchild ready', flush=True)\n"
                "time.sleep(60)\n")
            child = root / 'child.py'
            child.write_text(
                "import os,signal,subprocess,sys,time\n"
                "from pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                "subprocess.Popen([sys.executable,sys.argv[2],sys.argv[3]])\n"
                "time.sleep(60)\n")
            pid_files = [root / 'child.pid', root / 'grandchild.pid']
            try:
                result = run_llama(
                    [sys.executable, str(child), str(pid_files[0]), str(grandchild), str(pid_files[1])],
                    root / 'result', 10, 1.0)
                self.assertTrue(result['timed_out'])
                self.assertEqual(result['status'], 'error')
                self.assertEqual(result['finish_reason'], 'error')
                self.assertIn('grandchild ready', result['output'])
                for path in pid_files:
                    pid = int(path.read_text())
                    deadline = time.monotonic() + 2
                    while True:
                        status = subprocess.run(
                            ['ps', '-o', 'stat=', '-p', str(pid)],
                            capture_output=True, text=True).stdout.strip()
                        if not status or status.startswith('Z'):
                            break
                        if time.monotonic() >= deadline:
                            self.fail(f'inference descendant {pid} is still running: {status}')
                        time.sleep(0.02)
            finally:
                # Even the intentionally failing pre-fix test must not leak work.
                for path in pid_files:
                    if path.exists():
                        try:
                            os.kill(int(path.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_real_cli_cap_minus_one_log_is_not_eos(self):
        fixture = Path(__file__).resolve().parents[2] / "docs/benchmarks/2026-09-28-max32/llama-q8-256.stderr.txt"
        code = "import pathlib,sys; sys.stderr.write(pathlib.Path(sys.argv[1]).read_text())"
        with tempfile.TemporaryDirectory() as directory:
            result = run_llama([sys.executable, "-c", code, str(fixture)],
                               Path(directory) / "page", 256, 30)
        self.assertEqual(result["decode_steps"], 255)
        self.assertEqual(result["finish_reason"], "unknown")
        self.assertEqual(aggregate([result])["completed"], 0)
        self.assertIsNone(aggregate([result])["median_seconds"])

    def test_cli_finish_boundaries_include_prefill_first_token(self):
        for runs, limit, expected in [(0, 1, "unknown"), (254, 256, "eos"),
                                      (255, 256, "unknown"), (256, 256, "token_limit")]:
            with self.subTest(runs=runs, limit=limit), tempfile.TemporaryDirectory() as directory:
                code = f"import sys; sys.stderr.write('eval time = 1.0 ms / {runs} runs\\n')"
                result = run_llama([sys.executable, "-c", code], Path(directory) / "page", limit, 30)
                self.assertEqual(result["finish_reason"], expected)

    def test_terminal_eos_is_removed_only_with_matching_token_evidence(self):
        eos = "<｜end of sentence｜>"
        response = {"content": "quoted " + eos + " value" + eos,
                    "tokens": [17, 42, 1], "stop_type": "eos"}
        text, tokens = completion_text_without_eos(response, eos)
        self.assertEqual(text, "quoted " + eos + " value")
        self.assertEqual(tokens, [17, 42])
        self.assertEqual(response["tokens"], [17, 42, 1])
        for stopped, ids in [("limit", [17, 42, 1]), ("eos", [17, 42])]:
            other = dict(response, stop_type=stopped, tokens=ids)
            self.assertEqual(completion_text_without_eos(other, eos), (other["content"], ids))
        with self.assertRaises(ValueError):
            completion_text_without_eos(dict(response, content="missing marker"), eos)
        with self.assertRaises(ValueError):
            completion_text_without_eos(response, "")

    def test_thread_environment_is_validated_and_passed_to_child(self):
        original = os.environ.copy()
        with patch.dict(os.environ, {"VECLIB_MAXIMUM_THREADS": "99"}):
            environment = benchmark_environment(8, 1)
            self.assertEqual(os.environ["VECLIB_MAXIMUM_THREADS"], "99")
            self.assertEqual(environment["VECLIB_MAXIMUM_THREADS"], "1")
            self.assertEqual(environment["OMP_NUM_THREADS"], "8")
            with tempfile.TemporaryDirectory() as directory:
                code = "import os,json; print(json.dumps({k:os.environ[k] for k in ['VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']}))"
                result = run_llama([sys.executable, "-c", code], Path(directory) / "child", 10, 30,
                                   environment=environment)
            self.assertEqual(json.loads(result["output"]), {"VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "8"})
            self.assertEqual(result["thread_environment"], json.loads(result["output"]))
        self.assertEqual(dict(os.environ), original)
        for workers, blas in [(0, 1), (8, 0), (8, -1), (8, 9)]:
            with self.assertRaises(ValueError):
                benchmark_environment(workers, blas)

    def test_all_vision_batches_are_counted(self):
        log = ("mtmd batch encoding done in 7103 ms\n"
               "mtmd batch encoding done in 2059 ms\n"
               "mtmd batch encoding done in 2174 ms\n"
               "eval time = 1230.21 ms / 250 runs\n")
        code = f"import sys; sys.stderr.write({log!r})"
        with tempfile.TemporaryDirectory() as directory:
            result = run_llama([sys.executable, '-c', code], Path(directory) / "page", 2048, 30)
        self.assertEqual(result["vision_seconds"], 11.336)
        self.assertEqual(result["decode_seconds"], 1.23021)
        self.assertEqual(result["finish_reason"], "eos")

    def test_failure_and_truncation_cannot_improve_latency(self):
        records = [
            dict(status="ok", finish_reason="eos", wall_seconds=4.0),
            dict(status="ok", finish_reason="eos", wall_seconds=6.0),
            dict(status="error", finish_reason="error", wall_seconds=0.01),
            dict(status="ok", finish_reason="token_limit", wall_seconds=0.1),
        ]
        summary = aggregate(records)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["truncated"], 1)
        self.assertEqual(summary["median_seconds"], 5.0)
        self.assertEqual(summary["p95_seconds"], 6.0)

    def test_empty_and_failed_benchmark_has_no_throughput_claim(self):
        self.assertIsNone(aggregate([])["median_seconds"])
        self.assertIsNone(aggregate([dict(status="error")])["p95_seconds"])

    def test_nonfinite_and_negative_durations_rejected(self):
        for value in [float("nan"), float("inf"), -1.0]:
            with self.assertRaises(ValueError):
                aggregate([dict(status="ok", finish_reason="eos", wall_seconds=value)])

    def test_corrupt_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_image(image, "0" * 64)

    def test_character_edits_and_empty_reference(self):
        self.assertEqual(character_error_rate("ABCD", "ABCD"), 0.0)
        self.assertEqual(character_error_rate("ABCD", "ABXD"), 0.25)
        self.assertEqual(character_error_rate("中文识别", "中文识别"), 0.0)
        self.assertEqual(character_error_rate("abcd", "abc"), 0.25)
        self.assertEqual(character_error_rate("", ""), 0.0)
        with self.assertRaises(ValueError):
            character_error_rate("", "invented")

    def test_normalization_preserves_recognized_numbers(self):
        raw = "<|det|>text [0, 1, 20, 30]<|/det|>Invoice 123\nTotal 42.50"
        self.assertEqual(normalize_ocr(raw), "Invoice 123 Total 42.50")

    def test_html_table_cells_are_scored_as_recognized_text(self):
        raw = '<table><tr><td>A &amp; B</td><td>12.50</td></tr><tr><td>Total</td><td>12.50</td></tr></table>'
        self.assertEqual(normalize_ocr(raw), 'A & B 12.50 Total 12.50')


if __name__ == "__main__":
    unittest.main()
