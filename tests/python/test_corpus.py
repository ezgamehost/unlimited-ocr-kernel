"""Catch missing glyphs, clipped text and nondeterministic benchmark inputs."""
import tempfile
import unittest
from pathlib import Path

from tools.corpus import generate_corpus, require_glyphs


class CorpusTests(unittest.TestCase):
    def test_unknown_glyphs_are_rejected(self):
        with self.assertRaises(ValueError):
            require_glyphs(Path('/System/Library/Fonts/Menlo.ttc'), '\U0010ffff')

    def test_images_are_deterministic_and_have_ground_truth(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = generate_corpus(Path(a))
            second = generate_corpus(Path(b))
            self.assertEqual(len(first['images']), 4)
            self.assertEqual([x['sha256'] for x in first['images']],
                             [x['sha256'] for x in second['images']])
            for entry in first['images']:
                self.assertTrue(entry['text'].strip())
                self.assertLessEqual(entry['text_bounds'][3], entry['height'])
                self.assertLessEqual(entry['text_bounds'][2], entry['width'])


if __name__ == '__main__':
    unittest.main()
