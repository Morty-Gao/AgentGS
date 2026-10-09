"""One metadata authority for tools, planning, scientific contracts and indexes."""

import json
from importlib.resources import files
from typing import Any


def load_catalog() -> dict[str, Any]:
    """Read a fresh catalog without changing its content or declaration order."""
    return json.loads(files("agentgs.contracts").joinpath("catalog.json").read_text("utf-8"))


def get_tool_spec(name: str) -> dict[str, Any] | None:
    """Find one metadata record by its public tool name."""
    return next((row for row in load_catalog()["specs"] if row["Name"] == name), None)


def load_tool_scope() -> dict[str, Any]:
    """Load the public tool allowlist without any Skill/Capability declarations."""
    return json.loads(files("agentgs.contracts").joinpath("tool_scope.json").read_text("utf-8"))
