import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SkillMatch:
    skill_id: str
    score: float
    matched_terms: tuple[str, ...] = ()
    score_components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "score": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
            "score_components": {k: round(v, 4) for k, v in self.score_components.items()},
        }


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
