"""Expose the same declarative library to the planner, CLI and runtime."""

from agentgs.skills.definitions import load_skill_definitions
from agentgs.skills.implementations.declarative import DeclarativeSkill

# Keep the existing import path for callers that instantiate declarative Skills.
__all__ = ["DeclarativeSkill", "builtin_skills"]


def builtin_skills() -> dict[str, DeclarativeSkill]:
    return {row["id"]: DeclarativeSkill(row) for row in load_skill_definitions()}
