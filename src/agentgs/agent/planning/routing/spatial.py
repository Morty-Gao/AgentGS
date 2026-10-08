from __future__ import annotations

import re
from typing import Any

from agentgs.agent.planning.routing.common import _unique


def _deep_time_or_ccs(prefix: str, text: str, locator: dict[str, Any]) -> list[str]:
    normalized = re.sub("[-\\u2010-\\u2015_]+", " ", text.casefold())
    tools: list[str] = []
    if "convert" in normalized and "dms" in normalized:
        return [f"{prefix}DecimalToDMS"]
    if "distance" in normalized:
        tools.append(f"{prefix}HaversineDistance")
    if "bearing" in normalized:
        tools.append(f"{prefix}Bearing")
    if "bounding" in normalized or "bbox" in normalized:
        tools.append(f"{prefix}BBoxArea")
    has_nearest = any((token in normalized for token in ("nearest neighbour", "nearest neighbor")))
    has_buffer = "buffer" in normalized
    has_overlay = any((token in normalized for token in ("overlay", "join them to", "by zone")))
    has_batch = "batch" in normalized and any(
        (token in normalized for token in ("vector", "layer", "point", "process"))
    )
    explicit_standardize = any(
        (
            token in normalized
            for token in ("standardize", "standardise", "point layer", "quality control")
        )
    )
    explicit_validate = any(
        (token in normalized for token in ("validate", "quality control", "repair geometry"))
    )
    crs_repair_workflow = "crs" in normalized and any(
        (token in normalized for token in ("check", "mixed", "inconsisten", "4326/3857"))
    )
    if prefix == "CCS" and "crs" in normalized:
        tools.append("CCSCheckCRSConsistency")
    needs_vector_entry = has_nearest or has_buffer or has_overlay
    if explicit_standardize or needs_vector_entry or has_batch:
        tools.append(f"{prefix}StandardizeSpatialData")
    if has_batch:
        tools.append(f"{prefix}BatchVector")
    if explicit_validate or (needs_vector_entry and (not crs_repair_workflow)):
        tools.append(f"{prefix}ValidateSpatialData")
    if has_nearest:
        tools.append(f"{prefix}NearestNeighbor")
    if has_buffer:
        tools.append(f"{prefix}Buffer")
    if has_overlay:
        tools.append(f"{prefix}Overlay")
    explicit_zonal = any(
        (token in normalized for token in ("zonal", "raster summary", "raster statistics"))
    )
    summarized_by_zone = bool(
        re.search("\\bsummari[sz]e\\b[^.]{0,100}\\bby\\b[^.]{0,40}\\bzones?\\b", normalized)
    ) and (
        not bool(
            re.search(
                "\\bsummari[sz]e\\b[^.]{0,80}\\b(?:points?|locations?|records?)\\b", normalized
            )
        )
    )
    needs_zonal = explicit_zonal or summarized_by_zone
    explicit_interpolation = any(
        (token in normalized for token in ("interpolate", "idw", "continuous surface"))
    )
    implicit_interpolation = needs_zonal and "raster" in normalized
    needs_interpolation = explicit_interpolation or implicit_interpolation
    if needs_interpolation:
        tools.append(f"{prefix}InterpolateIDW")
    needs_raster_algebra = any(
        (token in normalized for token in ("raster algebra", "mask", "intensity", "index"))
    )
    if (
        prefix == "DeepTime"
        and needs_interpolation
        and ("age" in normalized)
        and ("continuous" in normalized)
    ):
        needs_raster_algebra = True
        needs_zonal = True
    if needs_raster_algebra:
        tools.append(f"{prefix}RasterAlgebra")
    if needs_zonal and "bounding" not in normalized:
        tools.append(f"{prefix}ZonalStatistics")
    if any((token in normalized for token in ("map", "render", "thematic"))):
        tools.append(f"{prefix}RenderMap")
    if not tools and locator:
        tools.append(f"{prefix}RenderMap")
    return _unique(tools)
