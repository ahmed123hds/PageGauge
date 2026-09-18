"""CPU checks for matched graph-pilot evidence reduction."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/mlsys2027/baselines_v1'))
import frontier_graph_pair as pair


class GraphPairTests(unittest.TestCase):
    def fixture(self, folder, validate):
        def write(name, value):
            (folder/name).write_text(json.dumps(value))
        write('tokens.json', {'ids': []})
        write('quality.json', {'test': True})
        write('manifest.json', {'batch': 4, 'context': 20480, 'decode_steps': 1536,
            'tokens_sha256': pair.base.sha256_file(folder/'tokens.json')})
        write('completion.json', {'return_code': 0, 'sampled_exclusivity_passed': True})
        row = {'steps': 1536, 'initial_state_validation': {'bitwise_initial_state_match': True} if validate else None}
        stats = {'served_plan_calls': 1536, 'graph_replays': 1536*32, 'missing_bank_count': 0,
                 'same_cache_oracle_calls': 352 if validate else 0}
        write('analysis.json', {'validation_only': validate, 'rows': [row] if validate else [row]*3,
            'warmup': None if validate else row, 'attention_graph_dispatch': [stats]*(1 if validate else 4),
            'quality_sha256': pair.base.sha256_file(folder/'quality.json') if validate else None})

    def test_validation_and_timing_contracts(self):
        for validate in (True, False):
            with self.subTest(validate=validate), tempfile.TemporaryDirectory() as tmp, patch.object(pair, 'verify'):
                folder = Path(tmp)
                self.fixture(folder, validate)
                pair.read_run(folder, validate)
                with self.assertRaises(ValueError):
                    pair.read_run(folder, not validate)
                result = json.loads((folder/'analysis.json').read_text())
                result['attention_graph_dispatch'][0]['missing_bank_count'] = 1
                (folder/'analysis.json').write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    pair.read_run(folder, validate)


if __name__ == '__main__':
    unittest.main()
