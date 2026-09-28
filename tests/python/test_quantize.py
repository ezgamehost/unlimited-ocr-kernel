import tempfile,unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file,load_file
from tools.quantize import quantize_blocks,quantize_file

class QuantizeTests(unittest.TestCase):
    def test_int4_simulation_uses_four_bit_levels_but_declares_int8_storage(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.safetensors';target=Path(d)/'simulation.safetensors'
            x=np.zeros((1,35),dtype=np.float32)
            x[0,:8]=[-7,-3.5,0,3.5,7,-.5,.5,1.5]
            save_file({'lm_head.weight':x},source)
            result=quantize_file(source,target,chunk_rows=1,bits=4)
            tensors=load_file(target)
            np.testing.assert_array_equal(tensors['lm_head.weight'][0,:8],[-7,-4,0,4,7,0,0,2])
            np.testing.assert_array_equal(tensors['lm_head.weight.q8_scales'],[[1.,0.]])
            self.assertEqual(tensors['lm_head.weight'].dtype,np.int8)
            self.assertEqual(result['simulated_weight_bits'],4)
            self.assertEqual(result['storage_bits'],8)
            from safetensors import safe_open
            with safe_open(target,framework='np') as reader:
                self.assertEqual(reader.metadata()['simulated_weight_bits'],'4')
                self.assertEqual(reader.metadata()['storage_bits'],'8')

    def test_zero_blocks_and_tails_roundtrip_with_bounded_error(self):
        x=np.arange(105,dtype=np.float32).reshape(3,35)/37-1
        x[0]=0
        q,s=quantize_blocks(x)
        self.assertEqual(q.dtype,np.int8);self.assertEqual(s.shape,(3,2))
        decoded=q.astype(np.float32)*np.repeat(s,32,axis=1)[:,:35]
        self.assertTrue(np.all(np.abs(decoded-x)<=np.repeat(s/2,32,axis=1)[:,:35]+1e-6))
        self.assertTrue(np.all(q[0]==0));self.assertTrue(np.all(s[0]==0))
        with self.assertRaises(ValueError):quantize_blocks(np.array([[np.nan]],dtype=np.float32))
    def test_export_keeps_router_and_norm_out_of_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.safetensors';target=Path(d)/'int8.safetensors'
            x=np.linspace(-1,1,105,dtype=np.float32).reshape(3,35)
            save_file({'lm_head.weight':x,'model.layers.1.mlp.gate.weight':np.ones((3,35),np.float32),'model.norm.weight':np.ones(35,np.float32)},source)
            before=source.read_bytes()
            result=quantize_file(source,target,chunk_rows=2)
            self.assertEqual(source.read_bytes(),before)
            tensors=load_file(target)
            self.assertEqual(set(tensors),{'lm_head.weight','lm_head.weight.q8_scales'})
            self.assertEqual(tensors['lm_head.weight'].dtype,np.int8)
            self.assertEqual(result['matrix_count'],1)
            with self.assertRaises(FileExistsError):quantize_file(source,target)

    def test_integer_source_matrix_is_rejected_without_publishing(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.safetensors';target=Path(d)/'int8.safetensors'
            save_file({'lm_head.weight':np.ones((2,3),dtype=np.int64)},source)
            with self.assertRaises(ValueError):quantize_file(source,target)
            self.assertFalse(target.exists())
