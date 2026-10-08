"""Expose the same declarative library to the planner, CLI and runtime."""

import json

from agentgs.skills.base import PromptSkill, SkillPreparation, SkillSpec
from agentgs.skills.definitions import load_skill_definitions


class DeclarativeSkill(PromptSkill):
    def __init__(self, definition):
        self.spec = SkillSpec(
            id=definition["id"],
            version=definition["version"],
            description=definition["description"],
            keywords=definition["keywords"],
            tools=definition["tool_candidates"],
            definition=definition,
        )

    def prompt(self):
        return json.dumps(self.spec.definition, ensure_ascii=False, indent=2)

    def validate_tools(self, names):
        # Candidates are alternatives; the declaration requires at least one.
        if not set(self.spec.tools).intersection(names):
            raise ValueError(f"No compatible tools available for Skill {self.spec.id}")

    def prepare(self, context):
        names = {spec.id for spec in context.tools.specs()}
        self.validate_tools(names)
        return SkillPreparation(
            prompt=self.prompt(),
            candidate_tools=[tool for tool in self.spec.tools if tool in names],
        )

    async def execute(self, call, context):
        # Shared capabilities can select a registered tool outside this Skill.
        return await context.tools.execute(call)


def builtin_skills():
    return {row["id"]: DeclarativeSkill(row) for row in load_skill_definitions()}
