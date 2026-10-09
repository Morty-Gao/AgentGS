"""Shared scientific tools and dataset adapter for this agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/benchmark"))
from _runtime import call_tool as _call_tool  # noqa: E402
from _runtime import catalog, load_questions, project_root, tool_workspace  # noqa: E402, F401


def call_tool(name: str, arguments: dict | None = None):
    """Return the structured tool result and keep an execution receipt."""
    return _call_tool(name, arguments, strict=False)
