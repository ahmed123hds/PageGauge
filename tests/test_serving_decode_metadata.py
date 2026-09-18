import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/serving_v1'))
from pagegauge_vllm.request_slots import RequestSlots
from pagegauge_vllm.partition import Policy
from pagegauge_vllm.decode_metadata import prepare_decode, commit_decode


class Metadata(unittest.TestCase):
    def setup_manager(self):
        m=RequestSlots(2,Policy(suffix_pages=0))
        a=m.admit('a',8192,range(512))
        b=m.admit('b',8207,range(1000,1513))
        return m,a,b

    def test_heterogeneous_reorder_and_aging(self):
        m,a,b=self.setup_manager()
        batch=prepare_decode(m,[(b,8207,range(1000,1513)),(a,8192,tuple(range(512))+(2000,))])
        self.assertEqual(batch.columns()['positions'],(8207,8192))
        self.assertEqual(batch.columns()['request_slots'],(1,0))
        self.assertEqual(batch.columns()['physical_token_slots'],(1512*16+15,2000*16))
        self.assertEqual(batch.rows[1].new_history_logical_pages,(464,))
        self.assertEqual(batch.rows[1].new_history_physical_pages,(464,))
        # The outgoing page and incoming page reuse the same ring slot:
        # the GPU must quantize the old page before writing the new token.
        self.assertEqual(batch.rows[1].new_history_exact_slots[0],batch.rows[1].exact_token_slot//16)
        self.assertEqual(m.state(a).current_tokens,8192)
        commit_decode(m,batch)
        self.assertEqual((m.state(a).current_tokens,m.state(b).current_tokens),(8193,8208))
        with self.assertRaises(ValueError): commit_decode(m,batch)

    def test_rejected_batch_does_not_partially_advance(self):
        m,a,b=self.setup_manager()
        with self.assertRaises(ValueError):
            prepare_decode(m,[(a,8192,tuple(range(512))+(2000,)),(b,8208,range(1000,1513))])
        self.assertEqual(m.state(a).current_tokens,8192)
        self.assertNotIn(2000,m.page_owner)

    def test_stale_generation_rejected(self):
        m,a,b=self.setup_manager()
        batch=prepare_decode(m,[(a,8192,tuple(range(512))+(2000,))])
        m.release(a); m.admit('a',8192,range(512))
        with self.assertRaises(ValueError): commit_decode(m,batch)

    def test_alias_introduced_after_preparation_rejected(self):
        m,a,b=self.setup_manager()
        batch=prepare_decode(m,[(a,8192,tuple(range(512))+(2000,))])
        m.append(b,8207,range(1000,1513))
        m.append(b,8208,tuple(range(1000,1513))+(2000,))
        with self.assertRaises(ValueError): commit_decode(m,batch)
        self.assertEqual(m.state(a).current_tokens,8192)

if __name__=='__main__': unittest.main()
