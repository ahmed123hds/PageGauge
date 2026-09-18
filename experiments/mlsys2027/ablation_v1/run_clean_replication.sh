#!/usr/bin/env bash
set -euo pipefail
root=/mnt/c/Users/noorg/Ahmed/page_gauge_protocol_v2_rtx5090
download_pid="${1:?provide verified downloader PID or none}"
contrast="${2:?provide reference_a0 or fi_a0}"
pilot="${3:?provide completed pilot directory}"
owned_download="$root/experiments/mlsys2027/tasks_v1/download_instruct.py"
is_owned_download() {
  [[ "$download_pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$download_pid/cmdline" ]] || return 1
  tr '\0' '\n' < "/proc/$download_pid/cmdline" | grep -Fxq "$owned_download"
}
resume_download() {
  if is_owned_download; then kill -CONT "$download_pid"; fi
}
if [[ "$download_pid" != none ]]; then
  is_owned_download || { printf 'Refusing to pause an unverified process\n' >&2; exit 1; }
  trap resume_download EXIT
  kill -STOP "$download_pid"
fi
bash "$root/experiments/mlsys2027/representation_v2/run.sh" ../ablation_v1/regional_replication --contrast "$contrast" --pilot "$pilot"
