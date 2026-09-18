"""Exact next-token alignment for native BOS and no-BOS book policies."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('book_tokens', ROOT/'experiments/mlsys2027/generalization_v1/book_tokens.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class BookTokens(unittest.TestCase):
    def test_bos_and_no_bos_next_labels(self):
        corpus = list(range(100, 120))
        for family, bos in [('mistral', 1), ('qwen3', None)]:
            ids, policy = MOD.window(corpus, 4, 3, family, bos)
            self.assertEqual(len(ids), 8)
            begin = policy['corpus_label_start_offset']
            end = policy['corpus_label_end_offset_exclusive']
            self.assertEqual(ids[5:8], corpus[begin:end])
            self.assertEqual(policy['corpus_tokens_per_request'], 7 if family == 'mistral' else 8)

    def test_no_short_book_replacement_or_unsupported_model(self):
        with self.assertRaises(ValueError):
            MOD.window([1, 2], 4, 3, 'qwen3', None)
        with self.assertRaises(ValueError):
            MOD.window(list(range(20)), 4, 3, 'mistral', None)
        with self.assertRaises(ValueError):
            MOD.window(list(range(20)), 4, 3, 'unknown', None)


if __name__ == '__main__':
    unittest.main()
