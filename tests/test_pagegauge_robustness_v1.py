"""CPU-only tests for proposed mathematics and fixed-input attribution."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1] / "experiments/mlsys2027/robustness_v1"
sys.path.insert(0, str(HERE))
import affine_reference as ref
from analyze import analyze_capture


class ReferenceTests(unittest.TestCase):
    def test_generalized_factorization_matches_explicit_reconstruction(self):
        rng = np.random.default_rng(19)
        for dim in (2, 4, 8):
            with self.subTest(dim=dim):
                q = rng.normal(size=(3, dim))
                zk, zv = (rng.integers(-127, 128, size=(11, dim)) for _ in range(2))
                sk, sv = (rng.uniform(0.005, 0.03, size=11) for _ in range(2))
                ak = np.eye(dim) + rng.normal(scale=0.05, size=(dim, dim))
                av = np.eye(dim) + rng.normal(scale=0.05, size=(dim, dim))
                ck, cv = (rng.normal(size=dim) for _ in range(2))
                k = ref.generalized_reconstruct(zk, sk, ak, ck)
                v = ref.generalized_reconstruct(zv, sv, av, cv)
                np.testing.assert_allclose(ref.generalized_factorized(q, zk, zv, sk, sv, ak, av, ck, cv),
                                           ref.attention(q, k, v), atol=1e-12, rtol=1e-12)

    def test_original_scalar_policy_is_special_case(self):
        q, z = np.ones((2, 4)), np.arange(24).reshape(6, 4)
        scale, center = np.full(6, 0.01), np.arange(4) * 0.1
        expected = ref.attention(q, z * 0.01 + center, z * 0.01 + center)
        np.testing.assert_allclose(ref.generalized_factorized(q, z, z, scale, scale, np.eye(4), np.eye(4), center, center), expected)

    def test_bad_metadata_rejected(self):
        with self.assertRaises(ValueError):
            ref.generalized_reconstruct(np.ones((3, 4)), np.ones(3), np.zeros((4, 4)), np.ones(4))
        with self.assertRaises(ValueError):
            ref.generalized_reconstruct(np.ones((3, 4)), -np.ones(3), np.eye(4), np.ones(4))

    def test_fixed_query_error_bound(self):
        rng = np.random.default_rng(23)
        for magnitude in (0, 0.001, 0.1, 10):
            q, k, v = rng.normal(size=(4, 8)), rng.normal(size=(13, 8)), rng.normal(size=(13, 6))
            result = ref.attention_error_bound(q, k, v, k + magnitude * rng.normal(size=k.shape),
                                               v + magnitude * rng.normal(size=v.shape))
            self.assertTrue(result["bound_satisfied_with_fp64_tolerance"])

    def test_common_key_shift_is_not_charged_as_distortion(self):
        rng = np.random.default_rng(29)
        q, k, v = rng.normal(size=(3, 4)), rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
        result = ref.attention_error_bound(q, k, v, k + np.array([2., -1., 3., 0.]), v)
        self.assertLess(max(result["epsilon_score_modulo_shift"]), 1e-14)
        self.assertLess(max(result["output_error_l2"]), 1e-14)

    def test_round_half_away_is_not_numpy_bankers_rounding(self):
        np.testing.assert_array_equal(ref.round_away(np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32)),
                                      [-3, -2, -1, 1, 2, 3])

    def test_exact_regions_and_complete_history(self):
        policy = {"page_size": 16, "exact_prefix_pages": 1, "exact_static_suffix_pages": 1, "exact_tail_tokens": 16}
        mask = ref.exact_mask(96, 64, policy)
        np.testing.assert_array_equal(np.flatnonzero(mask[::16]), [0, 3, 5])
        x = np.random.default_rng(31).normal(size=(97, 2, 4)).astype(np.float16)
        center = x[:64].astype(np.float32).mean(0).astype(np.float16)
        result = ref.mixed_reconstructions(x, center, 64, policy)
        self.assertTrue(result["exact_mask"][-1])
        np.testing.assert_array_equal(result["real"][result["exact_mask"]], result["centered"][result["exact_mask"]])

    def test_end_to_end_cpu_attribution_fixture_and_replay_comparison(self):
        rng = np.random.default_rng(37)
        q = rng.normal(size=(32, 128)).astype(np.float16)
        k, v = (rng.normal(size=(96, 8, 128)).astype(np.float16) for _ in range(2))
        centers = [x[:64].astype(np.float32).mean(0).astype(np.float16) for x in (k, v)]
        hf = np.concatenate([ref.attention(q[h*4:h*4+4], k[:, h], v[:, h]) for h in range(8)])
        plan = {"initial_context_tokens": 64, "policy": {"page_size": 16, "exact_prefix_pages": 1,
                                                            "exact_static_suffix_pages": 1, "exact_tail_tokens": 16}}
        with tempfile.TemporaryDirectory() as directory:
            path, replay = Path(directory) / "fixture.npz", Path(directory) / "replay.npz"
            np.savez(path, query=q, key=k, value=v, key_center=centers[0], value_center=centers[1], hf_attention_output=hf)
            np.savez(replay, factorized=hf, explicit=hf)
            result = analyze_capture(path, plan, replay)
        self.assertEqual(len(result["heads"]), 8)
        self.assertEqual(result["quantized_tokens"], 48)
        for head in result["heads"]:
            self.assertEqual(head["hf_sdpa_vs_original_fp64"]["max_abs"], 0)
            self.assertTrue(head["fixed_query_bound"]["bound_satisfied_with_fp64_tolerance"])
            self.assertEqual(head["factorized_vs_explicit_fp16"]["max_abs"], 0)
        json.dumps(result, allow_nan=False)

    def test_new_modules_do_not_import_gpu_libraries_at_top_level(self):
        for path in HERE.glob("*.py"):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                names = [x.name for x in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse({name.split(".")[0] for name in names} & {"torch", "flashinfer", "transformers"})


if __name__ == "__main__":
    unittest.main()
