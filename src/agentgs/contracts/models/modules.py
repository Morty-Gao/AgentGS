"""Composable interventions within the scientific planning runtime."""

from typing import Literal

from .base import Model

PlanningStructure = Literal["hierarchical", "flat_tools", "bipartite"]
ToolSelection = Literal["optimized", "first_valid"]
Ablation = Literal["A1", "A2", "A3", "A4", "A5", "A6", "BIPARTITE"]


class ModuleSettings(Model):
    planning_structure: PlanningStructure = "hierarchical"
    predecessor_expansion: bool = True
    capability_completion: bool = True
    directed_bridging: bool = True
    tool_selection: ToolSelection = "optimized"
    execution_recovery: bool = True


class ModuleOverrides(Model):
    planning_structure: PlanningStructure | None = None
    predecessor_expansion: bool | None = None
    capability_completion: bool | None = None
    directed_bridging: bool | None = None
    tool_selection: ToolSelection | None = None
    execution_recovery: bool | None = None


ABLATIONS: dict[str, dict] = {
    "A1": {"planning_structure": "flat_tools"},
    "A2": {"predecessor_expansion": False},
    "A3": {"capability_completion": False},
    "A4": {"directed_bridging": False},
    "A5": {"tool_selection": "first_valid"},
    "A6": {"execution_recovery": False},
    "BIPARTITE": {"planning_structure": "bipartite"},
}


def resolve_modules(base, ablations=(), overrides=None):
    if not isinstance(ablations, (list, tuple)):
        raise ValueError("ablations must be a list of preset names")
    values = base.model_dump()
    interventions = {}
    for name in ablations:
        if not isinstance(name, str) or name not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {name}")
        for key, value in ABLATIONS[name].items():
            if key in interventions and interventions[key] != value:
                raise ValueError(f"Conflicting ablations for {key}")
            interventions[key] = value
    explicit = ModuleOverrides.model_validate(overrides or {}).model_dump(exclude_none=True)
    for key, value in explicit.items():
        if key in interventions and interventions[key] != value:
            raise ValueError(f"Module override contradicts ablation for {key}")
    values.update(interventions)
    values.update(explicit)
    requested = dict(values)
    suppressed = {}
    if not values["capability_completion"] and values["directed_bridging"]:
        values["directed_bridging"] = False
        suppressed["directed_bridging"] = "capability_completion_disabled"
    return ModuleSettings.model_validate(values), {
        "ablations": sorted(set(ablations)),
        "requested": requested,
        "suppressed": suppressed,
    }
