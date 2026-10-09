#!/usr/bin/env bash
# Sample the GeoToolEval ablation subset on a running CPU pod (GPU=0).
# Launch this from the development machine. It does not start a pod and does
# not run the sampler locally when no CPU pod is available.
set -euo pipefail

LIST_WORKER="/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Script/list-worker.py"
PROJECT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"

cpu_pod_hosts() {
  "$LIST_WORKER" --json | python3 -c '
import json
import sys

workers = json.load(sys.stdin)
chosen = []
for worker in workers:
    status = str(worker.get("status") or "").casefold()
    gpu = worker.get("gpu")
    if status != "running" or gpu not in (0, None):
        continue
    ssh = str(worker.get("ssh") or "")
    host = ssh.split()[-1] if ssh.startswith("ssh ") else ""
    if not host.endswith(".pod@h.pjlab.org.cn"):
        continue
    cpu = worker.get("cpu")
    cpu_value = cpu if isinstance(cpu, (int, float)) else 0
    chosen.append((-cpu_value, str(worker.get("id") or ""), host))
for _cpu, _worker_id, host in sorted(chosen):
    print(host)
'
}

mapfile -t HOSTS < <(cpu_pod_hosts)
if ((${#HOSTS[@]} == 0)); then
  echo "No running CPU pod (GPU=0). Launch one and rerun. The sampler stays off this machine." >&2
  exit 1
fi

for host in "${HOSTS[@]}"; do
  echo "Sampling on CPU pod ${host}" >&2
  if ssh -o BatchMode=yes -o ConnectTimeout=20 "$host" bash -s -- "$@" <<EOF
set -euo pipefail
cd "$PROJECT"
exec python3 "$PROJECT/scripts/ablation/sample_subset.py" --size 60 --seed 0 "\$@"
EOF
  then
    exit 0
  fi
  echo "CPU pod ${host} failed. Trying the next CPU pod." >&2
done

echo "Every CPU pod failed. Launch a new CPU pod and rerun." >&2
exit 1
