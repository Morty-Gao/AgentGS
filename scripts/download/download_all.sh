#!/usr/bin/env bash
# Run with: bash scripts/download/download_all.sh [--verify-only]
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
while [[ ! -f "$repo_root/pyproject.toml" ]]; do
    parent="$(dirname -- "$repo_root")"
    [[ "$parent" != "$repo_root" ]] || { echo "pyproject.toml not found" >&2; exit 1; }
    repo_root="$parent"
done
python="$repo_root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    python="$repo_root/.venv/Scripts/python.exe"
fi
[[ -x "$python" ]] || { echo "Project Python not found; run uv sync first." >&2; exit 1; }
failed=0
for script in "$script_dir"/download_*.py; do
    "$python" -X utf8 "$script" "$@" || failed=$((failed + 1))
done
if (( failed > 0 )); then
    echo "$failed dataset commands failed; see errors above." >&2
    exit 1
fi
