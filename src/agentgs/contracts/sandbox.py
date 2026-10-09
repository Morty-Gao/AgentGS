"""Sandbox capabilities and shell arguments shared by agents and runtimes."""

from pathlib import Path
from typing import Any, Protocol

from pydantic import Field, field_validator

from agentgs.contracts.models import Model


class SandboxUnavailable(RuntimeError):
    pass


class BashInput(Model):
    command: str = Field(min_length=1, max_length=4000, strict=True)

    @field_validator("command")
    @classmethod
    def valid_command(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("bash command must be non-empty and contain no NUL bytes")
        return value


class DataSandbox(Protocol):
    root: Path

    async def describe_shell(self, data_root: Path) -> dict[str, Any]: ...
