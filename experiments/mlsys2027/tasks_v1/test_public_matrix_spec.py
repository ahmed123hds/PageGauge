import unittest
from public_matrix_spec import jobs, GROUPS
from task_contract import TASK_OUTPUT_LIMITS


class Tests(unittest.TestCase):
    def test_full_task_control_coverage(self):
        matrix = jobs()
        self.assertEqual(len(matrix), 80)
        self.assertEqual(len({j['id'] for j in matrix}), 80)
        for task in TASK_OUTPUT_LIMITS:
            for group in GROUPS:
                arms = [a for j in matrix if j['task'] == task and j['control_group'] == group['id'] for a in j['arms']]
                self.assertEqual(arms, group['arms'])
                self.assertEqual(arms.count('hf'), 1)


if __name__ == '__main__':
    unittest.main()
