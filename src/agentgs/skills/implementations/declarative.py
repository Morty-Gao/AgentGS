"""Runtime adapter for the frozen JSON Skill declarations."""

import json

from agentgs.contracts.models import ToolCall, ToolResult
from agentgs.skills.base import PromptSkill, SkillContext, SkillPreparation, SkillSpec


class DeclarativeSkill(PromptSkill):
    def __init__(self, definition: dict):
        self.spec = SkillSpec(
            id=definition["id"],
            version=definition["version"],
            description=definition["description"],
            keywords=definition["keywords"],
            tools=definition["tool_candidates"],
            definition=definition,
        )

    def prompt(self) -> str:
        return json.dumps(self.spec.definition, ensure_ascii=False, indent=2)

    def validate_tools(self, names: set[str]) -> None:
        # Candidates are alternatives; the declaration requires at least one.
        if not set(self.spec.tools).intersection(names):
            raise ValueError(f"No compatible tools available for Skill {self.spec.id}")

    def prepare(self, context: SkillContext) -> SkillPreparation:
        names = {spec.id for spec in context.tools.specs()}
        self.validate_tools(names)
        return SkillPreparation(
            prompt=self.prompt(),
            candidate_tools=[tool for tool in self.spec.tools if tool in names],
        )

    async def execute(self, call: ToolCall, context: SkillContext) -> ToolResult:
        # Shared capabilities can select a registered tool outside this Skill.
        return await context.tools.execute(call)
