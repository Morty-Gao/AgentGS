"""Scientific strategies and authored compositions over the registered tool scope."""

import json
from importlib.resources import files


def load_skill_definitions() -> list[dict]:
    # Decode per call so a planner cannot mutate another task's declarations.
    return [
        row
        for name in ("definitions.json", "compositions.json")
        for row in json.loads(files("agentgs.skills.resources").joinpath(name).read_text("utf-8"))
    ]


def load_capability_extensions() -> dict:
    return json.loads(
        files("agentgs.skills.resources").joinpath("capability_extensions.json").read_text("utf-8")
    )


def load_library_scope() -> dict:
    from agentgs.contracts.catalog import load_tool_scope

    return (
        json.loads(
            files("agentgs.skills.resources").joinpath("library_scope.json").read_text("utf-8")
        )
        | load_tool_scope()
    )
