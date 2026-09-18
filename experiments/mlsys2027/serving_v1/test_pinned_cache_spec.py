"""Run with the isolated pinned vLLM Python; no model or CUDA initialization."""
import unittest
from unittest.mock import patch
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from experiments.mlsys2027.serving_v1.pagegauge_vllm.cache_spec import packed_page_spec
from experiments.mlsys2027.serving_v1.pagegauge_vllm.cache_layout import Layout
from experiments.mlsys2027.serving_v1.pagegauge_vllm.cache_views import dense_pool_views, from_engine_layer, zero_physical_pages
from experiments.mlsys2027.serving_v1.pagegauge_vllm.sidecars import allocate_sidecars
from experiments.mlsys2027.serving_v1.pagegauge_vllm.partition import Policy


class PackedSpecTests(unittest.TestCase):
    def test_real_sidecar_storage_matches_reserved_bytes(self):
        layout = Layout(2, 32, 3, kv_heads=1, query_heads=2,
                        policy=Policy(1, 1, 2))
        accounting = layout.accounting()
        tensors = allocate_sidecars(layout, device='cpu',
            cache_budget_bytes=accounting['total_served_capacity_bytes'])
        self.assertEqual(sum(t.untyped_storage().nbytes() for t in tensors.values()),
                         accounting['reserved_sidecar_bytes'])
        self.assertEqual(len({t.data_ptr() for t in tensors.values()}), 5)
        self.assertEqual(tensors['exact_key'].shape, (2, 12, 16, 1, 128))
        self.assertEqual(tensors['output_center'].shape, (2, 3, 2, 128))
        for name, tensor in tensors.items():
            self.assertTrue(tensor.is_contiguous())
            self.assertEqual(tensor.dtype, torch.float16)
            self.assertEqual(torch.count_nonzero(tensor).item(), 0)
            self.assertEqual(tensor.numel()*tensor.element_size(),
                             accounting['tensors'][name])

    def test_sidecar_budget_rejected_before_any_allocation(self):
        layout = Layout(1, 32, 1)
        minimum = layout.accounting()['total_served_capacity_bytes']
        with patch('torch.zeros') as allocate:
            for budget in (minimum-1, 0):
                with self.assertRaises(MemoryError):
                    allocate_sidecars(layout, device='cpu', cache_budget_bytes=budget)
            for budget in (-1, True, float(minimum)):
                with self.assertRaises(ValueError):
                    allocate_sidecars(layout, device='cpu', cache_budget_bytes=budget)
            allocate.assert_not_called()

    def spec(self, **overrides):
        args = dict(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16)
        args.update(overrides)
        return FullAttentionSpec(**args)

    def test_page_accounting_matches_layout(self):
        for heads in (1, 4, 8):
            original = self.spec(num_kv_heads=heads)
            packed = packed_page_spec(original)
            expected = Layout(32, 256, 4, kv_heads=heads).accounting()
            self.assertEqual(packed.page_size_bytes, expected['per_layer_page_bytes'])
            self.assertEqual(packed.num_states, 1)
            self.assertEqual(packed.block_size, 16)
            self.assertEqual(original.dtype, torch.float16)
            self.assertEqual(original.page_size_bytes, 2*16*heads*128*2)

    def test_rejects_unsupported_specs(self):
        for overrides in (
            dict(block_size=32), dict(dtype=torch.bfloat16), dict(head_size=64),
            dict(sliding_window=4096), dict(attention_chunk_size=1024),
            dict(non_causal=True), dict(num_head_slots=8),
            dict(state_content_bytes=256), dict(tokens_per_state=2),
            dict(page_size_padded=65536),
            dict(kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD),
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                packed_page_spec(self.spec(**overrides))
        with self.assertRaises(ValueError):
            packed_page_spec(packed_page_spec(self.spec()))

    def test_no_cuda_initialization(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_pool_views_are_dense_disjoint_and_zero_copy(self):
        for pages in (1, 3, 17):
            for heads in (1, 4, 8):
                size = pages*packed_page_spec(self.spec(num_kv_heads=heads)).page_size_bytes
                raw = torch.zeros(size, dtype=torch.uint8)
                views = dense_pool_views(raw, pages, heads)
                occupied = 0
                for index, tensor in enumerate(views.values(), 1):
                    self.assertTrue(tensor.is_contiguous())
                    self.assertEqual(tensor.untyped_storage().data_ptr(), raw.data_ptr())
                    self.assertEqual(tensor.data_ptr(), raw.data_ptr()+occupied)
                    tensor.fill_(index)
                    occupied += tensor.numel()*tensor.element_size()
                self.assertEqual(occupied, size)
                for index, tensor in enumerate(views.values(), 1):
                    self.assertTrue(torch.all(tensor == index).item())
                self.assertEqual(views['key_scales'].stride(), (heads, 1))
                self.assertEqual(views['key_codes'].stride(), (16*heads*128, heads*128, 128, 1))

    def test_pool_view_rejects_wrong_storage(self):
        size = packed_page_spec(self.spec()).page_size_bytes
        for raw in (torch.zeros(size+1, dtype=torch.uint8),
                    torch.zeros(size, dtype=torch.float16),
                    torch.zeros(size*2, dtype=torch.uint8)[::2],
                    torch.zeros(size+1, dtype=torch.uint8)[1:]):
            with self.assertRaises(ValueError):
                dense_pool_views(raw, 1, 8)

    def test_engine_layer_offset_preserved(self):
        backing = torch.zeros((2, 3, 8, 1, 4100), dtype=torch.uint8)
        views = from_engine_layer(backing[1], 3, 8)
        self.assertEqual(views['key_codes'].data_ptr(), backing[1].data_ptr())
        for tensor in views.values():
            tensor.fill_(1)
        self.assertTrue(torch.all(backing[0] == 0).item())
        self.assertTrue(torch.any(backing[1] != 0).item())
        interleaved = torch.zeros((3, 2, 8, 1, 4100), dtype=torch.uint8)[:, 0]
        with self.assertRaises(ValueError):
            from_engine_layer(interleaved, 3, 8)

    def test_plane_zeroing_preserves_other_pages(self):
        raw = torch.empty(4*32800, dtype=torch.uint8)
        views = dense_pool_views(raw, 4, 8)
        for tensor in views.values():
            tensor.fill_(7)
        zero_physical_pages(views, [3, 1])
        for tensor in views.values():
            self.assertTrue(torch.all(tensor[[1, 3]] == 0).item())
            self.assertTrue(torch.all(tensor[[0, 2]] == 7).item())
        before = raw.clone()
        for pages in ([0, 4], [0, -1], [0, 0], [True]):
            with self.assertRaises(ValueError):
                zero_physical_pages(views, pages)
            self.assertTrue(torch.equal(raw, before))


if __name__ == '__main__':
    unittest.main()
