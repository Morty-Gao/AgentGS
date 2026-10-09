"""Planning policy and independent model, plan and tool recovery budgets."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from agentgs.contracts.models import Model, TaskInput


class AlgorithmSettings(Model):
    planning_mode: Literal["tool_only", "skill_guided", "data_only"] = "tool_only"
    allow_rule_planning: bool = False
    plan_attempts: int = Field(default=3, ge=1)
    tool_attempts: int = Field(default=3, ge=1)
    command_attempts: int = Field(default=12, ge=1)
    shared_bridge: bool = True
    rs_local_contracts: bool = True
    rs_explicit_references: bool = False
    geosparql_llm: bool = False
    geosparql_plan_timeout: float = 8
    model_attempts: int = Field(default=4, ge=1)
    argument_attempts: int = Field(default=3, ge=1)
    retry_base_seconds: float = 1
    retry_jitter_seconds: float = 0.35
    temperature: float | None = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_threads: int = 4
    embedding_batch_size: int = 8
    index_path: str = "data/geotoolkg"
    model_cache: str = "data/raw/models/fastembed"


def load_algorithm(root: Path):
    path = root / "config.toml"
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    return AlgorithmSettings.model_validate(raw.get("agent", {}))


def resolve_algorithm(options: AlgorithmSettings, task: TaskInput) -> AlgorithmSettings:
    """Resolve task overrides before building mode-specific runtime services."""
    values = options.model_dump()
    for name in (
        "planning_mode",
        "allow_rule_planning",
        "plan_attempts",
        "tool_attempts",
        "command_attempts",
    ):
        value = getattr(task, name, None)
        if value is not None:
            values[name] = value
    if values["planning_mode"] in {"tool_only", "data_only"}:
        values["shared_bridge"] = False
    if values["planning_mode"] == "data_only":
        values["allow_rule_planning"] = False
    return AlgorithmSettings.model_validate(values)
