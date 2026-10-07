from agentgs.skills.base import PromptSkill, SkillSpec


class DomainSkill(PromptSkill):
    spec = SkillSpec(
        id="geoscience-calculation",
        description="Compute WGS84 distances, bearings, areas and earthquake energy",
        keywords=["distance", "bearing", "area", "energy", "距离", "面积", "能量"],
        tools=["GeoHaversineDistance", "GeoMagnitudeToEnergy"],
    )
