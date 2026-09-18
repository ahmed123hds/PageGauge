from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/tasks_v1'))
from generation_contract import aligned_prefix, greedy_rollout


class GenerationContract(unittest.TestCase):
    def test_no_omitted_or_added_prompt_tokens(self):
        for length in (17, 31, 32, 33, 20480, 20481):
            context = aligned_prefix(length)
            self.assertEqual(context % 16, 0)
            self.assertTrue(1 <= length-context <= 16)
            prompt = list(range(length))
            observed = []
            def step(token, position):
                observed.append((token, position))
                return 70000+position
            result = greedy_rollout(prompt, context, 2, {2}, step)
            self.assertEqual(observed[:length-context], [(p, p) for p in range(context, length)])
            self.assertEqual(result['generated_ids'], [70000+length-1, 70000+length])
            self.assertEqual(observed[-1], (70000+length-1, length))

    def test_own_greedy_token_and_eos_not_reference_teacher_forcing(self):
        observed = []
        def step(token, position):
            observed.append((token, position))
            return 42 if position == 16 else 2
        result = greedy_rollout(list(range(17)), 16, 128, {2}, step)
        self.assertEqual(observed, [(16, 16), (42, 17)])
        self.assertEqual(result, {'generated_ids': [42, 2], 'stop_reason': 'eos', 'decoder_step_calls': 2})

    def test_invalid_partition_and_immediate_eos(self):
        with self.assertRaises(ValueError):
            aligned_prefix(16)
        with self.assertRaises(ValueError):
            greedy_rollout(list(range(32)), 32, 1, {2}, lambda *_: 2)
        result = greedy_rollout(list(range(32)), 16, 512, {2}, lambda *_: 2)
        self.assertEqual(result['generated_ids'], [2])
        self.assertEqual(result['decoder_step_calls'], 16)


if __name__ == '__main__':
    unittest.main()
