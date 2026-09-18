import io
import json
import unittest
import zipfile
from materialize_longbench import read_task
from prompt_accounting import summarize


class Tests(unittest.TestCase):
    def test_exact_archive_member_and_duplicate_ids(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as z:
            z.writestr('data/qasper.jsonl', json.dumps({'_id': 'one'})+'\n')
            z.writestr('data/hotpotqa.jsonl', 'not read')
        data.seek(0)
        with zipfile.ZipFile(data) as z:
            self.assertEqual(read_task(z, 'qasper'), [{'_id': 'one'}])
            with self.assertRaises(ValueError):
                read_task(z, 'qmsum')
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as z:
            z.writestr('data/qasper.jsonl', (json.dumps({'_id': 'same'})+'\n')*2)
        data.seek(0)
        with zipfile.ZipFile(data) as z:
            with self.assertRaises(ValueError):
                read_task(z, 'qasper')

    def test_central_truncation_survives_worker_preparation(self):
        fixtures = {'cases': [{'case_id': 'a', 'prompt_ids': [1, 2, 3],
                              'original_prompt_tokens': 10, 'removed_tokens': 7}]}
        self.assertEqual(summarize(fixtures)['removed_prompt_tokens'], 7)
        fixtures['cases'][0]['removed_tokens'] = 0
        with self.assertRaises(ValueError):
            summarize(fixtures)


if __name__ == '__main__':
    unittest.main()
