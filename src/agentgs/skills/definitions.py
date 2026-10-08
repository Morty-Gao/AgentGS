"""Frozen scientific declarations, independent of tool/backend availability.

These are the 24 definitions associated with the selected 101 GeoToolEval tools.
Loading declarations does not register a backend or imply migration completion.
"""

import json
from importlib.resources import files


def load_skill_definitions() -> list[dict]:
    # Decode per call so a planner cannot mutate another task's declarations.
    return json.loads(files("agentgs.skills").joinpath("definitions.json").read_text("utf-8"))


def load_capability_extensions() -> dict:
    return json.loads(
        files("agentgs.skills").joinpath("capability_extensions.json").read_text("utf-8")
    )


def load_tool_scope() -> dict:
    return json.loads(files("agentgs.skills").joinpath("tool_scope.json").read_text("utf-8"))
