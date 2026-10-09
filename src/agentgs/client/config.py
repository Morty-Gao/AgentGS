import tomllib
from pathlib import Path

from pydantic import ValidationError

from agentgs.client.models import LLMConfig
from agentgs.config.env import environment, interpolate


def load_llm(root: Path, path: Path | None = None) -> LLMConfig:
    config_path = path or root / "llm.toml"
    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    try:
        config = LLMConfig.model_validate(interpolate(raw, environment(root)))
    except ValidationError as exc:
        locations = [".".join(map(str, error["loc"])) for error in exc.errors()]
        raise ValueError("Invalid LLM configuration fields: " + ", ".join(locations)) from None
    if config.schema_version != 1:
        raise ValueError("Unsupported llm.toml schema version")
    for name, model in config.models.items():
        if model.provider not in config.providers:
            raise ValueError(f"Unknown provider for model {name}")
        if model.default_effort and model.default_effort not in model.supported_efforts:
            raise ValueError(f"Invalid default effort for model {name}")
        if model.max_output_tokens >= model.context_window:
            raise ValueError(f"Output budget must be smaller than context for {name}")
    if len(config.models) != 1:
        raise ValueError("llm.toml must define exactly one model")
    config.configured_model()
    return config
