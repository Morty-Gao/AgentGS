"""Metadata observations and constraint diagnostics; no invented validation.

The baseline observes the presence of metadata. It does not prove numeric CRS,
extent, resolution or unit compatibility. Preserve that algorithm and report
its scope honestly rather than adding a new scientific admission policy.
"""

import re

from agentgs.agent.retrieval.scored_skills import split_resource

ACTIONS = {
    "crs": "reproject",
    "spatial_extent": "clip_or_reproject",
    "spatial_resolution": "resample",
    "temporal": "align_or_select_nearest",
    "format": "convert_format",
    "bands": "select_or_pair_bands",
    "variables": "select_variable",
    "units": "convert_units",
    "nodata": "mask_or_fill_nodata",
}
ALIASES = {
    "source_crs": "crs",
    "target_crs": "crs",
    "bbox": "spatial_extent",
    "resolution": "spatial_resolution",
    "date": "temporal",
    "year": "temporal",
    "time": "temporal",
}


def observations(query):
    _, resource = split_resource(query)
    values = {}
    if isinstance(resource, dict):
        locator = resource.get("source_locator")
        locator = locator if isinstance(locator, dict) else resource
        for key in [
            "crs",
            "source_crs",
            "target_crs",
            "bbox",
            "resolution",
            "date",
            "year",
            "time",
            "bands",
            "variables",
            "units",
            "nodata",
            "format",
        ]:
            if key in locator:
                values[ALIASES.get(key, key)] = locator[key]
        assets = resource.get("input_assets", [])
        if assets:
            values.setdefault("format", assets)
    text = query.casefold()
    epsg = re.findall(r"epsg[: ]?\d+", text, re.I)
    if epsg:
        values.setdefault("crs", epsg)
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:m|meter|km)\b", text):
        values.setdefault("spatial_resolution", "mentioned")
    if re.search(r"\b(?:19|20)\d{2}\b|\b\d+\s*(?:kyr|year|day|month)", text):
        values.setdefault("temporal", "mentioned")
    return values


def check_constraints(skill, query):
    values = observations(query)
    checks, recovery = [], []
    for name, spec in skill.get("geo_constraints", {}).items():
        if not isinstance(spec, dict):
            continue
        observed = values.get(name)
        status = (
            "satisfied"
            if observed is not None
            else "unknown"
            if spec.get("required")
            else "assumed"
        )
        auto_fix = spec.get("auto_fix")
        auto_fix = auto_fix if isinstance(auto_fix, dict) else {}
        if status == "unknown" and auto_fix.get("enabled"):
            status = "repairable"
            if auto_fix.get("preferred_skill"):
                recovery.append(auto_fix["preferred_skill"])
        checks.append(
            {
                "constraint": name,
                "status": status,
                "observed": observed,
                "rule": spec,
                "repair_action": ACTIONS.get(name),
                "repairable": bool(
                    auto_fix.get("enabled") or name in {"format", "bands", "variables", "units"}
                ),
            }
        )
    statuses = {row["status"] for row in checks}
    return {
        "skill_id": skill["id"],
        "status": "repairable"
        if "repairable" in statuses
        else "needs_metadata"
        if "unknown" in statuses
        else "satisfied_or_assumed",
        "checks": checks,
        "observations": values,
        "recovery_skills": list(dict.fromkeys(recovery)),
    }
