"""Opt-in sampled WSL GPU ownership checks; never changes historical NVML gates.

Root /proc scan covers device handles in this WSL kernel, including other users.
Windows NVML detects native compute clients; PID 4 is the WSL aggregate, not a
Linux PID. This is sampled context ownership, not continuous utilization proof.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CONTRACT = 'wsl_dxg_all_users_plus_windows_nvml_v1'
HOST_SMI = '/mnt/c/Windows/System32/nvidia-smi.exe'
WSL = '/mnt/c/Windows/System32/wsl.exe'


def scan_dxg():
    if os.geteuid() != 0:
        raise RuntimeError('Full /proc coverage requires a read-only root scan')
    pids = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process/'fd').iterdir():
                try:
                    if os.readlink(fd) == '/dev/dxg':
                        pids.append(int(process.name))
                        break
                except FileNotFoundError:
                    continue  # Descriptor closed concurrently.
        except FileNotFoundError:
            continue  # Process exited concurrently; permission errors are fatal.
    return sorted(pids)


def parse_host(raw, uuid):
    pids = []
    for row in csv.reader(line for line in raw.splitlines() if line.strip()):
        if len(row) != 2 or not row[1].strip().isdigit():
            raise ValueError('Unavailable Windows compute-process telemetry')
        if row[0].strip() == uuid:
            pids.append(int(row[1]))
    return sorted(set(pids))


def assess(dxg_pids, host_pids, child_pid):
    if not all(type(p) is int and p > 0 for p in dxg_pids+host_pids):
        raise ValueError('Invalid process identifiers')
    unexpected = sorted(set(dxg_pids)-{child_pid})
    native = sorted(set(host_pids)-{4})
    return {'pids': sorted(set(dxg_pids)), 'unexpected_pids': unexpected,
        'windows_compute_pids': host_pids, 'unexpected_windows_compute_pids': native,
        **({'error': 'Unexpected native Windows GPU process'} if native else {})}


def snapshot(gpu_uuid, child_pid):
    # No service, permission, environment, driver or process state is modified.
    scan = subprocess.run([WSL, '-d', 'Ubuntu', '-u', 'root', '--', '/usr/bin/python3',
        str(Path(__file__).resolve()), '--scan'], check=True, capture_output=True,
        text=True, timeout=15)
    host = subprocess.run([HOST_SMI, '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'monitor_contract': CONTRACT,
        **assess(json.loads(scan.stdout), parse_host(host.stdout, gpu_uuid), child_pid)}


def run_process(previous, command, output_dir, block, gpu_uuid):
    old = previous.process_snapshot
    try:
        previous.process_snapshot = snapshot
        result = previous.run_process(command, output_dir, block, gpu_uuid)
    finally:
        previous.process_snapshot = old
    result['monitor_contract'] = CONTRACT
    result['monitor_scope'] = __doc__
    return result


if __name__ == '__main__':
    if sys.argv[1:] != ['--scan']:
        raise ValueError('Only --scan is supported')
    print(json.dumps(scan_dxg()))
