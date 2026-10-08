#!/usr/bin/env bash
# On the Linux server: bash scripts/deploy/bootstrap.sh
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
cd -- "$project_dir"
command -v uv >/dev/null || { echo 'Install uv on the server, then rerun this script.' >&2; exit 1; }
command -v python3 >/dev/null || { echo 'python3 is required to start installation.' >&2; exit 1; }
python3 scripts/tools/install_reproduction.py --python 3.12
tools/envs/reproduction/bin/python scripts/tools/install_reproduction.py --verify
tools/envs/reproduction/bin/python scripts/deploy/smoke.py
echo 'Server environment and two offline tool tasks verified; no model or benchmark was run.'
echo 'For model tasks, configure .env and run: tools/envs/reproduction/bin/python -m agentgs config validate'
