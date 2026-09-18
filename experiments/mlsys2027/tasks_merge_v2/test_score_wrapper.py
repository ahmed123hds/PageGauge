"""Wrapper contract tests; original metric correctness is tested separately."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import score_matrix as scoring


class ScoreWrapperTests(unittest.TestCase):
    def test_provenance_and_receipt_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / 'job'
            job.mkdir()
            manifest = job / 'manifest.json'
            manifest.write_text(json.dumps({'merge_center_amendment': {}}))
            receipt = job / 'amendment_execution.json'
            valid = dict(version=2, post_exposure_correction=True,
                         manifest_sha256=scoring.sha(manifest))
            receipt.write_text(json.dumps(valid))
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(freeze_sha256='frozen',
                amendment_sha256='amended', jobs=[dict(directory='job')])))
            output = root / 'report.json'
            argv = ['score_matrix', '--plan', str(plan), '--output', str(output),
                    '--freeze', str(root/'freeze'), '--freeze-sha256', 'frozen',
                    '--materialized', str(root/'data'), '--metrics', str(root/'metrics')]

            def fake_metrics():
                target = Path(sys.argv[sys.argv.index('--output') + 1])
                target.write_text(json.dumps({'sentinel_metric': 42}))

            evidence = dict(post_exposure_correction=True, independent_test_claim=False)
            with patch.object(scoring, 'audit', return_value=evidence), \
                    patch.object(scoring.original, 'main', side_effect=fake_metrics) as metrics, \
                    patch.object(sys, 'argv', argv):
                # Reject missing, mismatched, or misleading receipts before metrics run.
                for key, bad in [('version', 1), ('post_exposure_correction', False),
                                 ('manifest_sha256', 'wrong')]:
                    receipt.write_text(json.dumps(dict(valid, **{key: bad})))
                    with self.assertRaises(ValueError):
                        scoring.main()
                    metrics.assert_not_called()
                receipt.unlink()
                with self.assertRaises(FileNotFoundError):
                    scoring.main()
                metrics.assert_not_called()
                receipt.write_text(json.dumps(valid))
                scoring.main()
                self.assertIs(sys.argv, argv)
                result = json.loads(output.read_text())
                self.assertEqual(result['sentinel_metric'], 42)
                provenance = result['evaluation_provenance']
                self.assertIs(provenance['post_exposure_correction'], True)
                self.assertIs(provenance['independent_test_claim'], False)
                self.assertEqual(provenance['original_metrics_report_sha256'],
                                 scoring.sha(root/'report.json.metrics.json'))
                with self.assertRaises(ValueError):
                    scoring.main()
                self.assertEqual(metrics.call_count, 1)


if __name__ == '__main__':
    unittest.main()
