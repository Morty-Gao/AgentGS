from agentgs.skills.base import PromptSkill, SkillSpec


class DomainSkill(PromptSkill):
    spec = SkillSpec(
        id="earthquake-catalog-analysis",
        description="Query earthquake records in the frozen USGS catalog",
        keywords=["earthquake", "catalog", "usgs"],
        tools=["DeepEarthQueryUSGSCatalog", "DeepEarthGetUSGSEvent"],
    )
