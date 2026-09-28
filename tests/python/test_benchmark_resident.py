import hashlib
import tempfile
from types import SimpleNamespace
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tools.benchmark_resident import LineReader, StreamAccumulator, schedule, wait_child, run


class ResidentBenchmarkTests(unittest.TestCase):
    def test_precision_reads_tensor_storage_despite_misleading_filename_and_file_type(self):
        import gguf
        import numpy as np
        import tools.benchmark_resident as resident
        with tempfile.TemporaryDirectory() as directory:
            for kind in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.Q8_0):
                path = Path(directory) / 'model-Q8.gguf'
                writer = gguf.GGUFWriter(path, arch='test')
                writer.add_uint32('general.file_type', 7)
                data = gguf.quants.quantize(np.ones((2, 32), dtype=np.float32), kind)
                writer.add_tensor('matrix', data, raw_dtype=kind)
                writer.add_tensor('norm', np.ones(32, dtype=np.float32))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                self.assertTrue(hasattr(resident, 'gguf_storage'), 'GGUF precision inspection missing')
                actual = resident.gguf_storage(path)
                self.assertEqual(actual['tensor_types'], {kind.name: 1, 'F32': 1})
                self.assertEqual(actual['elements_by_type'], {kind.name: 64, 'F32': 32})
                self.assertIn(kind.name, actual['label'])

    def test_native_driver_records_resources_and_repeated_independent_requests(self):
        self.native_driver_case()

    def test_native_driver_accepts_declared_single_view_and_rejects_mismatch(self):
        self.native_driver_case(273, 273)
        self.native_driver_case(273, 1513, expect_error=True)

    def native_driver_case(self, expected_tokens=None, actual_tokens=1513, expect_error=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            image.write_bytes(b'fixture bytes')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'images': [{'id': 'sample', 'path': 'image.png',
                'text': 'OK', 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}]}))
            if expected_tokens is not None:
                data = json.loads(manifest.read_text())
                data['images'][0]['expected_image_tokens'] = expected_tokens
                manifest.write_text(json.dumps(data))
            binary = root / 'fake-native'
            binary.write_text('#!' + sys.executable + '\n' + """
import json, sys
print(json.dumps({'event': 'loaded', 'load_seconds': 0.0}), file=sys.stderr, flush=True)
for line in sys.stdin:
    request = json.loads(line)
    assert request['prompt'] == '<image>\\ndocument parsing.'
    result = dict(text='OK', token_ids=[17, 42], finish_reason='eos', precision='fixture',
                  image_tokens=EXPECTED_TOKENS, timings=dict(first_token_seconds=.01, decode_seconds=.01,
                  decode_steps=2, vision_seconds=.01, prefill_seconds=.01))
    print(json.dumps(dict(status='ok', result=result)), flush=True)
""".replace('EXPECTED_TOKENS', str(actual_tokens)))
            binary.chmod(0o755)
            args = SimpleNamespace(engine='native', binary=binary, model=root,
                    int8_weights=None, int8_activations=False, threads=1, blas_threads=1,
                    manifest=manifest, output=root / 'result', warmups=1, repetitions=1,
                    seed=42, max_tokens=8192, timeout=1, startup_timeout=.5)
            if expect_error:
                with self.assertRaisesRegex(RuntimeError, 'unexpected image token count'):
                    run(args)
                return
            summary = run(args)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['per_image']['sample']['completed'], 1)
            process = json.loads((args.output / 'process.json').read_text())
            self.assertGreater(process['resources']['peak_rss_bytes'], 0)
            self.assertEqual(process['resources']['exit_code'], 0)
            expected = 1513 if expected_tokens is None else expected_tokens
            self.assertEqual(process['expected_tokens_by_image']['sample'],
                             {'image_tokens': expected, 'prefix_tokens': expected + 5})
            self.assertEqual(process['prefix_tokens'], expected + 5)

    def test_server_timing_rejects_wrong_prefix_and_cache_reuse(self):
        import tools.benchmark_resident as resident
        self.assertTrue(hasattr(resident, 'validate_server_timing'))
        resident.validate_server_timing({'cache_n': 0, 'prompt_n': 278}, 278)
        for timing in ({'cache_n': 0, 'prompt_n': 1518},
                       {'cache_n': 1, 'prompt_n': 278}):
            with self.assertRaises(ValueError):
                resident.validate_server_timing(timing, 278)

    def test_actual_stream_preserves_tokens_and_requires_terminal_event(self):
        state = StreamAccumulator()
        lines = Path('tests/fixtures/llama_completion_stream.sse').read_bytes().splitlines(True)
        for line in lines:
            state.push(line)
        final = state.finish()
        self.assertEqual(len(final['tokens']), 251)
        self.assertEqual(final['tokens'][-1], 1)
        self.assertEqual(final['stop_type'], 'eos')
        self.assertEqual(final['timings']['cache_n'], 0)
        partial = StreamAccumulator()
        partial.push(lines[0])
        with self.assertRaises(ValueError):
            partial.finish()
        combined = StreamAccumulator()
        combined.push(b'data: {"content":"ab","tokens":[5,6],"stop":false}\n')
        combined.push(b'data: {"content":"","tokens":[],"stop":true,"tokens_predicted":2}\n')
        self.assertEqual(combined.finish()['tokens'], [5, 6])

    def test_schedule_has_exact_per_image_warmup_and_measurement_counts(self):
        jobs = schedule(['a', 'b', 'c'], 2, 10, 42)
        self.assertEqual(jobs, schedule(['a', 'b', 'c'], 2, 10, 42))
        for image in ['a', 'b', 'c']:
            self.assertEqual(sum(j['image_id'] == image and j['warmup'] for j in jobs), 2)
            self.assertEqual(sum(j['image_id'] == image and not j['warmup'] for j in jobs), 10)
        self.assertTrue(all(j['warmup'] for j in jobs[:6]))
        self.assertTrue(all(not j['warmup'] for j in jobs[6:]))

    def test_reader_retains_partial_bytes_and_reports_timeout(self):
        read_fd, write_fd = os.pipe()
        log = io.BytesIO()
        try:
            reader = LineReader(read_fd, log)
            os.write(write_fd, b'first\npartial')
            self.assertEqual(reader.line(.2), b'first\n')
            with self.assertRaises(TimeoutError):
                reader.line(.01)
            self.assertEqual(log.getvalue(), b'first\npartial')
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_wait_collects_exact_child_peak_memory_and_cleans_timeouts(self):
        child = subprocess.Popen([sys.executable, '-c', "x=bytearray(b'x'*(32<<20)); assert x[-1]==120"],
                                 start_new_session=True)
        result = wait_child(child, 10)
        self.assertEqual(result['exit_code'], 0)
        self.assertGreater(result['peak_rss_bytes'], 32 << 20)
        self.assertEqual(child.returncode, 0)
        sleepy = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                  start_new_session=True)
        stopped = wait_child(sleepy, .02)
        self.assertTrue(stopped['forced_kill'])
        self.assertIsNotNone(sleepy.returncode)
