"""Explicit versions of the historical algorithm switches and budgets."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from agentgs.contracts.models import Model


class AlgorithmSettings(Model):
    planning_mode: Literal["tool_only", "skill_guided"] = "tool_only"
    require_llm: bool = False
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
    index_path: str = "data/indexes/geotoolkg"
    model_cache: str = "data/models/fastembed"


def load_algorithm(root: Path):
    path = root / "agent.toml"
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    return AlgorithmSettings.model_validate(raw.get("agent", {}))
