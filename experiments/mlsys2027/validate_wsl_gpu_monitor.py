"""Bounded live positive/negative controls for the opt-in WSL ownership monitor."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'diagnostics'))
import mlsys_rtx5090_entry as base
import wsl_gpu_monitor as monitor


def main():
    device = base.idle_preflight(0)[-1]['uuid']
    out = ROOT/'results/mlsys2027_monitor_v1'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    code = "import torch,sys,os; x=torch.ones(1024,device=0); torch.cuda.synchronize(); print(os.getpid(),flush=True); sys.stdin.readline()"
    children = []
    rows = {}
    try:
        rows['idle'] = monitor.snapshot(device, -1)
        for label in ('own', 'foreign'):
            child = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True)
            children.append(child)
            if int(child.stdout.readline()) != child.pid:
                raise ValueError('Probe ownership mismatch')
            rows[label] = monitor.snapshot(device, children[0].pid)
        if rows['idle']['pids'] or rows['idle'].get('error'):
            raise ValueError('Not idle')
        if rows['own']['pids'] != [children[0].pid] or rows['own'].get('error'):
            raise ValueError('Owned context not independently visible')
        if rows['foreign']['unexpected_pids'] != [children[1].pid]:
            raise ValueError('Foreign Linux context not rejected')
        if rows['own']['windows_compute_pids'] != [4]:
            raise ValueError('WSL aggregate visibility not validated')
        base.atomic_json(out/'analysis.json', {'passed': True, 'observations': rows,
            'source_sha256': {str(p): base.sha256_file(p) for p in (Path(__file__), Path(monitor.__file__))},
            'scope': 'Tiny allocation ownership controls only, not performance or quality evidence'})
        print('Live ownership controls PASS: '+str(out), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.stdin.write('\n'); child.stdin.flush()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate(); child.wait(timeout=10)
            child.stdin.close(); child.stdout.close()


if __name__ == '__main__': main()
