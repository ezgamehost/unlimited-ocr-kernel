"""Catch unverified downloads and loss of valid files after a failed download."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.fetch_model import download


class FetchTests(unittest.TestCase):
    def test_valid_download_and_idempotent_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            source.write_bytes(b"abc")
            digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            download(source.as_uri(), target, digest)
            self.assertEqual(target.read_bytes(), b"abc")
            source.unlink()
            download(source.as_uri(), target, digest)
            self.assertEqual(target.read_bytes(), b"abc")

    def test_hash_mismatch_never_publishes_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            source.write_bytes(b"corrupt")
            target.write_bytes(b"existing")
            with self.assertRaises(ValueError):
                download(source.as_uri(), target, hashlib.sha256(b"expected").hexdigest())
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertFalse(target.with_suffix(".part").exists())


if __name__ == "__main__":
    unittest.main()
