"""Unified-launcher integration plus outstanding KIVI2 resident cohort check."""
from datetime import datetime, timezone
from pathlib import Path
import uuid
from synthetic_generation import ROOT
from build_job_matrix import build
from run_frozen_jobs import run


def main():
    results = ROOT/'results/mlsys2027_tasks_v1'
    source = results/'kivi4_cohort_smoke_20260910T093433Z_95c55494'
    specifications = [{'id': backend, 'source': str(source), 'fixtures': str(source/'fixtures.json'),
        'worker': str(Path(__file__).with_name('native_task_worker.py')),
        'python': '/home/anonymous/pagegauge_baselines/kivi_mistral_sm120_env/bin/python',
        'control_group': 'mistral_native_4362', 'updates': {'task_backend': backend}}
        for backend in ('hf', 'kivi_int2')]
    root = results/('matrix_integration_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    plan = build(root, specifications, 'synthetic_integration',
                 {'scope': 'Saved long+short prompts; one HF and KIVI2 resident process; no public data.'})
    print(str(plan), flush=True)
    run(plan)


if __name__ == '__main__':
    main()
