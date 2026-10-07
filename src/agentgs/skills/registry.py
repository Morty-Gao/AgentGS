from agentgs.skills.earthquake_catalog.skill import DomainSkill as Earthquakes
from agentgs.skills.geoscience.skill import DomainSkill as Geoscience


def builtin_skills():
    return {skill.spec.id: skill for skill in [Earthquakes(), Geoscience()]}
