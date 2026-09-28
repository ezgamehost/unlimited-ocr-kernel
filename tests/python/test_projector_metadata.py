import tempfile,unittest
from pathlib import Path
import numpy as np
import gguf
from tools.projector_metadata import copy_with_crop_limit

class ProjectorMetadataTests(unittest.TestCase):
    def test_add_crop_limit_preserves_original_and_every_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.gguf'; target=Path(directory)/'target.gguf'
            writer=gguf.GGUFWriter(source,arch='clip')
            writer.add_string('test.note','unchanged')
            writer.add_tensor('test.weight',np.arange(24,dtype=np.float16).reshape(3,8))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            original=source.read_bytes()
            result=copy_with_crop_limit(source,target,32)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(result['max_tiles'],32)
            self.assertEqual(result['tensor_count'],1)
            reader=gguf.GGUFReader(target)
            self.assertEqual(reader.get_field('clip.vision.preproc_max_tiles').contents(),32)
            with self.assertRaises(FileExistsError):copy_with_crop_limit(source,target,32)
    def test_reject_invalid_limit_and_source_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.gguf';path.write_bytes(b'original')
            for cap in [0,1,257]:
                with self.assertRaises(ValueError):copy_with_crop_limit(path,Path(directory)/'b.gguf',cap)
            with self.assertRaises(ValueError):copy_with_crop_limit(path,path,32)
            self.assertEqual(path.read_bytes(),b'original')
