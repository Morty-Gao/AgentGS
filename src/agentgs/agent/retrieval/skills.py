import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SkillMatch:
    skill_id: str
    score: float


class SkillRetriever(Protocol):
    def retrieve(self, query: str, limit: int = 3) -> list[SkillMatch]: ...


class KeywordSkillRetriever:
    def __init__(self, registry):
        self.registry = registry

    def retrieve(self, query, limit=3):
        normalized = query.casefold()
        tokens = set(re.findall(r"\w+", normalized))
        matches = []
        for skill in self.registry.values():
            score = sum(
                1
                for term in skill.spec.keywords
                if term.casefold() in tokens or term.casefold() in normalized
            )
            if score:
                matches.append(SkillMatch(skill.spec.id, float(score)))
        return sorted(matches, key=lambda x: (-x.score, x.skill_id))[:limit]
