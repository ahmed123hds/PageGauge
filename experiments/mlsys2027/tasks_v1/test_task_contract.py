import unittest
from task_contract import prepare, validate_result


class Tests(unittest.TestCase):
    def test_middle_truncation_and_budget(self):
        c = prepare('gov_report', list(range(40000)), 32768)
        self.assertEqual(len(c.prompt_ids), 30720)
        self.assertEqual(c.prompt_ids[:3], (0, 1, 2))
        self.assertEqual(c.prompt_ids[-3:], (39997, 39998, 39999))
        self.assertLessEqual(len(c.prompt_ids)+c.max_new_tokens, c.context_limit)
        self.assertFalse(c.fp16_fallback)

    def test_prompt_remainder_and_fallback(self):
        for n in (1, 16, 17, 4096, 4097, 4112):
            c = prepare('qasper', [1]*n, 32768)
            if c.prefill_tokens:
                self.assertTrue(1 <= n-c.prefill_tokens <= 16)
            self.assertEqual(c.fp16_fallback, n <= 4096)

    def test_termination(self):
        c = prepare('hotpotqa', [1]*8193, 32768)
        validate_result(c, [5, 2], 'eos', [2])
        validate_result(c, [5]*32, 'max_new_tokens', [2])
        for ids, reason in [([2, 5], 'eos'), ([5], 'eos'), ([5], 'max_new_tokens')]:
            with self.assertRaises(ValueError):
                validate_result(c, ids, reason, [2])


if __name__ == '__main__':
    unittest.main()
