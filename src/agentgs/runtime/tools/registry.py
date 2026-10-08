from dataclasses import dataclass
from typing import Callable

from pydantic import Field

from agentgs.contracts.models import Model, ToolSpec
from agentgs.runtime.tools.domains.geosparql import SnapshotQuery
from agentgs.skills.definitions import load_tool_scope


class CodeInput(Model):
    code: str = Field(min_length=1, max_length=100000)


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    input_type: type
    handler: Callable | None
    source_type: str = "raw_compute"


def builtin_tools():
    from typing import Any

    from pydantic import RootModel

    from agentgs.contracts.catalog import load_catalog

    class Arguments(RootModel[dict[str, Any]]):
        pass

    allowed = set(load_tool_scope()["tool_ids"])
    result = {}
    for row in load_catalog()["specs"]:
        name = row["Name"]
        if name not in allowed:
            continue
        schema = SnapshotQuery if name == "GeoBigDataExecuteGeoSPARQL" else Arguments
        lookup = name.startswith(
            ("DeepEarthQueryLITHO", "DeepEarthQuerySEMUCB", "DeepEarthSample", "Ocean")
        ) or name in {
            "EnergyCapacityValue",
            "EnergyNearestFacility",
            "EnergyCountryRenewableSummary",
            "EnergyTransitionMetrics",
        }
        result[name] = RegisteredTool(
            ToolSpec(
                id=name,
                description=row["Function"],
                input_schema=schema.model_json_schema(),
                category=row["Category"],
                legacy_input=row["Input"],
                legacy_output=row["Output"],
            ),
            schema,
            None,
            source_type="derived_lookup"
            if lookup or name == "GeoBigDataExecuteGeoSPARQL"
            else "raw_compute",
        )
    return result


def internal_operations():
    """Infrastructure operations for explicit requests; excluded from tool retrieval."""
    return {
        "ExecutePython": RegisteredTool(
            ToolSpec(
                id="ExecutePython",
                description="Run Python in the configured code sandbox",
                input_schema=CodeInput.model_json_schema(),
                effects="write",
            ),
            CodeInput,
            None,
        )
    }
