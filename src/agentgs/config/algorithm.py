"""Planning policy and independent model, plan and tool recovery budgets."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agentgs.contracts.models import Model, TaskInput
from agentgs.contracts.models.modules import ModuleSettings, resolve_modules

MODULE_ALIASES = {
    "predecessor_expansion": "predecessor_expansion",
    "capability_completion": "capability_completion",
    "shared_bridge": "directed_bridging",
    "tool_selection": "tool_selection",
    "execution_recovery": "execution_recovery",
}


def module_value(name, value):
    return "optimized" if name == "tool_selection" and value == "model" else value


class AlgorithmSettings(Model):
    planning_mode: Literal["tool_only", "skill_guided", "data_only"] = "tool_only"
    allow_rule_planning: bool = False
    plan_attempts: int = Field(default=3, ge=1)
    tool_attempts: int = Field(default=3, ge=1)
    command_attempts: int = Field(default=20, ge=1)
    shared_bridge: bool = True
    modules: ModuleSettings = Field(default_factory=ModuleSettings)
    module_resolution: dict = Field(default_factory=dict)
    rs_local_contracts: bool = True
    rs_explicit_references: bool = False
    geosparql_llm: bool = False
    geosparql_plan_timeout: float = 8
    model_attempts: int = Field(default=4, ge=1)
    argument_attempts: int = Field(default=3, ge=1)
    predecessor_expansion: bool = True
    capability_completion: bool = True
    tool_selection: Literal["model", "first_valid"] = "model"
    execution_recovery: bool = True
    candidate_pool: Literal["retrieved", "all_registered"] = "retrieved"
    routing_hints: bool = True
    binding_assistance: bool = True
    bridge_diagnostics: bool = False
    benchmark_scope: Literal["full", "common_data"] = "full"
    max_model_calls: int = Field(default=200, ge=1)
    max_total_tokens: int = Field(default=262144, ge=1)
    retry_base_seconds: float = 1
    retry_jitter_seconds: float = 0.35
    temperature: float | None = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_threads: int = 4
    embedding_batch_size: int = 8
    index_path: str = "data/geotoolkg"
    model_cache: str = "model/models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"

    @model_validator(mode="after")
    def synchronize_module_aliases(self):
        if self.module_resolution:
            required = {"requested", "ablations", "suppressed", "active"}
            if set(self.module_resolution) != required:
                raise ValueError("Incomplete module resolution record")
            requested = ModuleSettings.model_validate(self.module_resolution["requested"])
            effective, record = resolve_modules(requested, self.module_resolution["ablations"])
            record["active"] = self.planning_mode == "skill_guided"
            if not record["active"]:
                effective = ModuleSettings(directed_bridging=False)
                record["suppressed"] = {"modules": "planning_mode_has_no_module_interventions"}
            if record != self.module_resolution or effective != self.modules:
                raise ValueError("Module resolution record disagrees with the effective policy")
            for alias, key in MODULE_ALIASES.items():
                if module_value(alias, getattr(self, alias)) != getattr(effective, key):
                    raise ValueError(f"Module resolution record disagrees with {alias}")
            return self
        values = self.modules.model_dump()
        for alias, key in MODULE_ALIASES.items():
            if alias in self.model_fields_set:
                value = module_value(alias, getattr(self, alias))
                if key in self.modules.model_fields_set and values[key] != value:
                    raise ValueError(f"Conflicting config settings: {alias} and modules.{key}")
                values[key] = value
        self.modules = ModuleSettings.model_validate(values)
        for alias, key in MODULE_ALIASES.items():
            value = values[key]
            setattr(
                self,
                alias,
                "model" if alias == "tool_selection" and value == "optimized" else value,
            )
        return self


def load_algorithm(root: Path):
    path = root / "config.toml"
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    if "module_resolution" in raw.get("agent", {}):
        raise ValueError("module_resolution is runtime-generated metadata, not a project setting")
    return AlgorithmSettings.model_validate(raw.get("agent", {}))


def resolve_algorithm(options: AlgorithmSettings, task: TaskInput) -> AlgorithmSettings:
    """Resolve task overrides before building mode-specific runtime services."""
    options = AlgorithmSettings.model_validate(options.model_dump())
    values = options.model_dump()
    for name in (
        "planning_mode",
        "allow_rule_planning",
        "plan_attempts",
        "tool_attempts",
        "command_attempts",
        "predecessor_expansion",
        "capability_completion",
        "shared_bridge",
        "tool_selection",
        "execution_recovery",
        "candidate_pool",
        "routing_hints",
        "binding_assistance",
        "bridge_diagnostics",
        "max_total_tokens",
    ):
        value = getattr(task, name, None)
        if value is not None:
            values[name] = value
    base_values = dict(options.module_resolution.get("requested", options.modules.model_dump()))
    overrides = task.modules.model_dump(exclude_none=True) if task.modules else {}
    for alias, key in MODULE_ALIASES.items():
        value = getattr(task, alias, None)
        if value is not None:
            value = module_value(alias, value)
            if key in overrides and overrides[key] != value:
                raise ValueError(f"Conflicting task settings: {alias} and modules.{key}")
            overrides[key] = value
    modules, resolution = resolve_modules(
        ModuleSettings.model_validate(base_values), task.ablations, overrides
    )
    if values["planning_mode"] != "skill_guided":
        if task.ablations or task.modules is not None or overrides:
            raise ValueError("Module ablations require planning_mode='skill_guided'")
        resolution["active"] = False
        resolution["suppressed"] = {"modules": "planning_mode_has_no_module_interventions"}
        modules = ModuleSettings(directed_bridging=False)
        for alias, key in MODULE_ALIASES.items():
            value = getattr(modules, key)
            values[alias] = "model" if alias == "tool_selection" and value == "optimized" else value
    else:
        resolution["active"] = True
        if modules != ModuleSettings() and values["allow_rule_planning"]:
            raise ValueError(
                "Module ablations require allow_rule_planning=false so heuristic routes cannot bypass interventions"
            )
        for alias, key in MODULE_ALIASES.items():
            value = getattr(modules, key)
            values[alias] = "model" if alias == "tool_selection" and value == "optimized" else value
    values["modules"] = modules
    values["module_resolution"] = resolution
    if values["planning_mode"] == "data_only":
        values["allow_rule_planning"] = False
    return AlgorithmSettings.model_validate(values)
