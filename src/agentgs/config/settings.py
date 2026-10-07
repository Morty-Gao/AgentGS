import tomllib
from pathlib import Path

from pydantic import Field

from agentgs.config.env import environment, interpolate
from agentgs.contracts.models import Model


class RuntimeSettings(Model):
    max_concurrent_tasks: int = Field(default=2, ge=1, le=128)
    default_timeout_seconds: float = Field(default=300, gt=0)
    default_rss_limit_mb: int = Field(default=1024, ge=64)
    max_total_rss_mb: int = Field(default=4096, ge=64)
    code_timeout_seconds: float = Field(default=20, gt=0, le=300)
    sandbox: str = "auto"
    allow_proot: bool = False


def load_runtime(root: Path) -> RuntimeSettings:
    path = root / "runtime.toml"
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    return RuntimeSettings.model_validate(interpolate(raw.get("runtime", {}), environment(root)))
