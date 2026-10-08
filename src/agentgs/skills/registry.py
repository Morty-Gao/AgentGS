from agentgs.skills.earthquake_catalog.skill import DomainSkill as Earthquakes
from agentgs.skills.semantic_geospatial_query.skill import DomainSkill as GeoQuestions


def builtin_skills():
    return {skill.spec.id: skill for skill in [Earthquakes(), GeoQuestions()]}
