from agentgs.skills.base import PromptSkill, SkillSpec


class DomainSkill(PromptSkill):
    spec = SkillSpec(
        id="semantic-geospatial-query",
        description="Retrieve frozen GeoQuestions/YAGO2geo materialized query results",
        keywords=[
            "geoquestions",
            "geosparql",
            "yago2geo",
            "lake",
            "forest",
            "river",
            "park",
            "county",
        ],
        tools=["GeoBigDataExecuteGeoSPARQL"],
    )
