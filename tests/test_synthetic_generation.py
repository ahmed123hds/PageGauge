from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/mlsys2027/tasks_v1'))
from synthetic_generation import make_case, score


class SyntheticGeneration(unittest.TestCase):
    def test_answer_extraction_does_not_cherry_pick_later_numbers(self):
        self.assertEqual(score(' 123456\n', '123456'),
            {'exact_stripped_match': True, 'first_six_digit_code_match': True})
        self.assertFalse(score('654321 then 123456', '123456')['first_six_digit_code_match'])
        self.assertFalse(score('11234567', '123456')['first_six_digit_code_match'])

    def test_deterministic_unique_needle_and_prompt_budget(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.message = messages[0]['content']
                return list(range(len(self.message.split())+8))
        tokenizer = Tokenizer()
        a = make_case(tokenizer, 8192, .5, 2026091000, 'qwen3')
        b = make_case(tokenizer, 8192, .5, 2026091000, 'qwen3')
        self.assertEqual(a, b)
        self.assertLessEqual(len(a['prompt_ids']), 8192)
        self.assertEqual(a['prompt_text'].count('archive ORCHID has code'), 1)
        self.assertEqual(a['prompt_text'].count(a['expected_code']), 1)
        self.assertEqual(a['needle_record_index'], int(a['record_count']*.5))


if __name__ == '__main__': unittest.main()
