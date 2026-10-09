"""Skill selection policy, distinct from ranking and capability expansion."""

from agentgs.agent.retrieval.scored_skills import WeightedSkillRetriever, extension_eligible
from agentgs.agent.planning.capabilities import GENERIC_OPERATIONS


def select_skill_ids(definitions, query: str, inferred_tools=(), *, mapping_extensions=True):
    skills = {skill["id"]: skill for skill in definitions}
    inferred = list(dict.fromkeys(inferred_tools))
    tool_skills = {}
    for skill in skills.values():
        for tool in skill.get("tool_candidates", []):
            tool_skills.setdefault(tool, skill["id"])
    selected = []
    for tool in inferred:
        skill_id = tool_skills.get(tool)
        if (
            skill_id
            and skill_id not in selected
            and extension_eligible(
                skills[skill_id], query, inferred, mapping_extensions=mapping_extensions
            )
        ):
            selected.append(skill_id)
    candidates = WeightedSkillRetriever(
        skills.values(), mapping_extensions=mapping_extensions
    ).retrieve(query, tool_names=inferred)
    candidates = [match for match in candidates if (
        set(match.matched_terms) - GENERIC_OPERATIONS
        or any(match.score_components.get(key, 0) > 0 for key in ("tool_hint", "domain", "data_type"))
    )]
    threshold = candidates[0].score * 0.55 if candidates else 0.0
    for match in candidates:
        if match.score < threshold and selected:
            continue
        if match.skill_id not in selected:
            selected.append(match.skill_id)
        # Preserve the historical soft cap: inferred hints are not truncated.
        if len(selected) >= 3:
            break
    return selected
