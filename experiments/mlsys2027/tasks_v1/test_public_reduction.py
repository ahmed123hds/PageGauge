import unittest
from unittest.mock import patch
from public_result_reduction import reduce_complete
from public_matrix_spec import GROUPS
from task_contract import TASK_OUTPUT_LIMITS


class Tests(unittest.TestCase):
    def test_controls_and_preprocessing_are_separate(self):
        references, fixtures, predictions = {}, {}, {}
        for task in TASK_OUTPUT_LIMITS:
            references[task] = [{'id': task, 'task': task, 'answers': ['synthetic']}]
            for group in GROUPS:
                fixtures[task, group['model']] = {'cases': [{'case_id': task, 'prompt_ids': [1, 2],
                    'original_prompt_tokens': 5, 'removed_tokens': 3}]}
                predictions[task, group['id']] = []
        calls = []
        def scorer(task, refs, rows, arms, *rest):
            calls.append((task, tuple(arms)))
            return {'rows': {a: {} for a in arms}}
        with patch('public_result_reduction.score_cohort', side_effect=scorer):
            result = reduce_complete(references, fixtures, predictions, None)
        self.assertEqual(len(calls), 40)
        self.assertEqual(result['tasks']['qasper']['mistral_nsn']['prompt_accounting']['removed_prompt_tokens'], 3)
        del predictions['qasper', 'mistral_nsn']
        with self.assertRaises(ValueError):
            reduce_complete(references, fixtures, predictions, None)


if __name__ == '__main__':
    unittest.main()
