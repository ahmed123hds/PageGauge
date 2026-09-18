import unittest
from types import SimpleNamespace
from cohort_scoring import score_cohort


class Tests(unittest.TestCase):
    def test_pairing_and_missing_results(self):
        examples = [{'id': 'a', 'answers': ['yes']}, {'id': 'b', 'answers': ['no']}]
        rows = [{'id': e['id'], 'backend': b, 'text': e['answers'][0], 'status': 'complete',
                 'fallback': False, 'executed_backend': b} for e in examples for b in ('hf', 'pg')]
        metrics = SimpleNamespace(qa_f1_score=lambda p, a, **kw: float(p == a))
        result = score_cohort('qasper', examples, rows, ['hf', 'pg'], metrics, 7, 100)
        self.assertEqual(result['rows']['pg']['paired_example_95_interval'], [0, 0])
        self.assertEqual(result['rows']['pg']['score_percent'], 100)
        with self.assertRaises(ValueError):
            score_cohort('qasper', examples, rows[:-1], ['hf', 'pg'], metrics, 7, 100)
        rows[-1]['status'] = 'failed'
        with self.assertRaises(ValueError):
            score_cohort('qasper', examples, rows, ['hf', 'pg'], metrics, 7, 100)


if __name__ == '__main__':
    unittest.main()
