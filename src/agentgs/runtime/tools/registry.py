from dataclasses import dataclass
from functools import partial
from typing import Any, Awaitable, Callable

from pydantic import Field

from agentgs.contracts.catalog import load_tool_scope
from agentgs.contracts.models import Model, ToolSpec
from agentgs.contracts.sandbox import BashInput
from agentgs.contracts.tool_inputs import tool_input_models
from agentgs.runtime.tools.domains.geosparql import SnapshotQuery, replay_query


class CodeInput(Model):
    code: str = Field(min_length=1, max_length=100000)


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    input_type: type
    handler: Callable[..., Any | Awaitable[Any]] | None
    source_type: str = "raw_compute"
    process_safe: bool = False


def builtin_tools():
    from agentgs.contracts.catalog import load_catalog
    from agentgs.runtime.tools.dispatch import execute_domain

    allowed = set(load_tool_scope()["tool_ids"])
    models = dict(tool_input_models())
    models["GeoBigDataExecuteGeoSPARQL"] = SnapshotQuery
    if set(models) != allowed:
        raise RuntimeError(f"Tool argument coverage mismatch: {set(models) ^ allowed}")
    result = {}
    for row in load_catalog()["specs"]:
        name = row["Name"]
        if name not in allowed:
            continue
        schema = models[name]
        lookup = name.startswith(
            ("DeepEarthQueryLITHO", "DeepEarthQuerySEMUCB", "DeepEarthSample")
        ) or name in {
            "EnergyCapacityValue",
            "EnergyNearestFacility",
            "EnergyNearestMineralSite",
            "EnergyCountryRenewableSummary",
            "EnergyTransitionMetrics",
            "OceanMonthlyGridValue",
            "OceanProfileValue",
            "OceanCruiseQC",
            "OceanAnnualCarbonSummary",
            "OceanVerticalProfileDifference",
            "OceanRegionalMapSummary",
            "OceanSurfaceDeepMatch",
            "OceanProfileDeduplication",
        }
        result[name] = RegisteredTool(
            ToolSpec(
                id=name,
                description=row["Function"],
                input_schema=schema.model_json_schema(),
                category=row["Category"],
                input_description=row["Input"],
                output_description=row["Output"],
            ),
            schema,
            replay_query if name == "GeoBigDataExecuteGeoSPARQL" else partial(execute_domain, name),
            source_type="derived_lookup"
            if lookup or name == "GeoBigDataExecuteGeoSPARQL"
            else "raw_compute",
            process_safe=True,
        )
    return result


def internal_operations():
    """Infrastructure operations for explicit requests; excluded from tool retrieval."""
    return {
        "bash": RegisteredTool(
            ToolSpec(
                id="bash",
                description="Explore read-only /data with persistent writable /work; no network",
                input_schema=BashInput.model_json_schema(),
                effects="write",
            ),
            BashInput,
            None,
        ),
        "ExecutePython": RegisteredTool(
            ToolSpec(
                id="ExecutePython",
                description="Run Python in the configured code sandbox",
                input_schema=CodeInput.model_json_schema(),
                effects="write",
            ),
            CodeInput,
            None,
        ),
    }
