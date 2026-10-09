#!/usr/bin/env bash
# Discover an existing CPU worker; never allocate a pod or execute locally.
set -euo pipefail
PROJECT="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
LIST_WORKER="/mnt/shared-storage-user/dnacoding/wangzhongqi/Code/Script/list-worker.py"
script="${1:?Usage: on_cpu_worker.sh prepare.py|run.py|summarize.py [arguments...]}"
shift
case "$script" in prepare.py|run.py|summarize.py) ;; *) echo "Unknown experiment entry" >&2; exit 2 ;; esac
for discovery in 1 2; do
  mapfile -t hosts < <("$LIST_WORKER" --json | UV_OFFLINE=1 uv run --project "$PROJECT" --no-sync python -c '
import json,sys
rows=[r for r in json.load(sys.stdin) if r.get("gpu")==0 and str(r.get("status", "")).lower()=="running"]
for row in sorted(rows, key=lambda r: (r.get("cpu") or 0, r["id"])):
    parts=str(row.get("ssh", "")).split()
    if len(parts)==2 and parts[0]=="ssh" and parts[1].endswith(".pod@h.pjlab.org.cn"):
        print(parts[1])
')
  for host in "${hosts[@]}"; do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" true; then
      printf -v remote 'bash -s -- %q %q' "$PROJECT" "$script"
      for argument in "$@"; do printf -v quoted ' %q' "$argument"; remote+="$quoted"; done
      exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" "$remote" <<'EOF'
set -euo pipefail
project="$1"
script="$2"
shift 2
cd "$project"
# Explicit labpon-equivalent environment for non-interactive network calls.
export http_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export no_proxy=127.0.0.1,localhost,10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn
export NO_PROXY="$no_proxy"
export UV_OFFLINE=1
exec uv run --no-sync python "scripts/ablation/$script" "$@"
EOF
    fi
  done
done
echo "No reachable running CPU worker. Start one yourself and rerun; no local fallback." >&2
exit 1
