"""Reference adapter must preserve pixel quantization and original prompt IDs."""
import tempfile
import unittest
from pathlib import Path
import os

from PIL import Image
from tools.reference import prepare_inputs, export_vision, export_decoder


class ReferenceTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('UOCR_REAL_MODEL'), 'requires original model weights')
    def test_vision_fixture_has_real_finite_projected_tokens(self):
        import torch
        tensors, metadata = prepare_inputs(Path('artifacts/upstream'), Path('/tmp/unlimited-ocr-baseline/baidu.png'), 'base')
        features = export_vision(Path('artifacts/upstream'), tensors, metadata, torch.float32)
        self.assertEqual(list(features['image_embeddings'].shape), [273, 1280])
        self.assertEqual(list(features['global.sam'].shape), [1, 1024, 16, 16])
        self.assertTrue(torch.isfinite(features['image_embeddings']).all().item())
        self.assertGreater(features['image_embeddings'].std().item(), 0.01)

    @unittest.skipUnless(os.environ.get('UOCR_DECODER_REFERENCE'), 'requires full decoder reference pass')
    def test_decoder_trace_reaches_ring_wrap_and_returns_logits(self):
        import torch
        image = torch.zeros(273, 1280)
        metadata = {'token_ids': [0] + [128815] * 273 + [123], 'image_tokens': 273}
        result = export_decoder(Path('artifacts/upstream'), image, metadata, steps=300)
        self.assertEqual(list(result['decode_token_ids'].shape), [300])
        self.assertEqual(result['logits'].shape[1], 129280)
        self.assertTrue(torch.isfinite(result['logits']).all().item())
        self.assertEqual(result['layer.11.hidden'].shape[1], 575)

    def test_small_black_image_has_273_image_tokens_and_bos(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'black.png'
            Image.new('RGB', (16, 16), (0, 0, 0)).save(path)
            tensors, metadata = prepare_inputs(Path('artifacts/upstream'), path, 'base')
            self.assertEqual(list(tensors['global'].shape), [1, 3, 1024, 1024])
            self.assertEqual(tensors['global'].min().item(), -1.0)
            self.assertEqual(tensors['global'].max().item(), -1.0)
            self.assertEqual(metadata['image_tokens'], 273)
            self.assertEqual(metadata['token_ids'][0], 0)
            self.assertEqual(metadata['token_ids'][1:274], [128815] * 273)
            self.assertEqual(metadata['crop_grid'], [1, 1])


if __name__ == '__main__':
    unittest.main()
