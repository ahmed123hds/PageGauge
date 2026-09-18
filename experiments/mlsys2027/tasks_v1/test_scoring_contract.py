import unittest
from types import SimpleNamespace
from scoring_contract import example_score, aggregate


class Tests(unittest.TestCase):
    def test_first_line_and_multiple_references(self):
        metrics = SimpleNamespace(qa_f1_score=lambda p, a, **kw: float(p == a))
        self.assertEqual(example_score('triviaqa', '\nright\nextra', ['wrong', 'right'], metrics), 1)
        self.assertEqual(example_score('qasper', 'right\nextra', ['right'], metrics), 0)

    def test_complete_cohort_and_fallback(self):
        rows = [{'id': 'a', 'score': 1, 'fallback': True}, {'id': 'b', 'score': 0, 'fallback': False}]
        self.assertEqual(aggregate(rows, ['a', 'b']), {'score_percent': 50, 'examples': 2, 'fallback_fraction': .5})
        with self.assertRaises(ValueError):
            aggregate(rows[:1], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
