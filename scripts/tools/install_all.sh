#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/../.."
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" "$SCRIPT_DIR/install_node.py" "$@"
"$PYTHON" "$SCRIPT_DIR/install_bwrap.py" "$@"
# PRoot is optional and must also be explicitly enabled with --allow-proot.
