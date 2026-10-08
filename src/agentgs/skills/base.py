from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from agentgs.contracts.models import (
    ArtifactStore,
    EventSink,
    Model,
    ToolCall,
    ToolExecutor,
    ToolResult,
)


class SkillSpec(Model):
    id: str
    version: str = "1.0"
    description: str
    keywords: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    definition: dict = Field(default_factory=dict)


@dataclass
class SkillContext:
    tools: ToolExecutor
    artifacts: ArtifactStore
    events: EventSink


class SkillPreparation(Model):
    prompt: str
    candidate_tools: list[str] = Field(default_factory=list)


class Skill(Protocol):
    spec: SkillSpec

    def prompt(self) -> str: ...
    def prepare(self, context: SkillContext) -> SkillPreparation: ...
    async def execute(self, call: ToolCall, context: SkillContext) -> ToolResult: ...
    def validate_tools(self, names: set[str]) -> None: ...


class PromptSkill:
    spec: SkillSpec

    def prepare(self, context: SkillContext) -> SkillPreparation:
        self.validate_tools({spec.id for spec in context.tools.specs()})
        return SkillPreparation(prompt=self.prompt(), candidate_tools=self.spec.tools)

    async def execute(self, call: ToolCall, context: SkillContext) -> ToolResult:
        if call.tool not in self.spec.tools:
            raise ValueError(f"Tool {call.tool} is outside Skill {self.spec.id}")
        return await context.tools.execute(call)

    def prompt(self):
        from importlib.resources import files

        return files(self.__class__.__module__.rsplit(".", 1)[0]).joinpath("prompt.md").read_text()

    def validate_tools(self, names):
        missing = set(self.spec.tools) - names
        if missing:
            raise ValueError(f"Missing skill tools: {sorted(missing)}")
