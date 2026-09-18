import unittest
from task_contract import prepare
from short_prompt_fallback import run


class Tests(unittest.TestCase):
    def test_independent_calls_and_disclosure(self):
        calls = []
        def generate(prompt, eos, limit):
            calls.append((prompt, limit))
            return {'generated_ids': [10+len(calls), 2], 'stop_reason': 'eos'}
        c = prepare('qasper', [1, 5, 6], 32768)
        arms = run(c, [2], generate)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({tuple(a['generated_ids']) for a in arms.values()}), 3)
        self.assertTrue(all(a['executed_backend'] == 'native_hf_fp16' and a['quantized_tokens_served'] == 0 for a in arms.values()))
        with self.assertRaises(ValueError):
            run(prepare('qasper', [1]*8193, 32768), [2], generate)


if __name__ == '__main__':
    unittest.main()
