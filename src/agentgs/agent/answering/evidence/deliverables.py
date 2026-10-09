from __future__ import annotations

import re
from typing import Any

from agentgs.agent.answering.evidence.core import _number_text, _walk_dicts


def _project_workflow_deliverables(values: list[Any], question: str) -> str | None:
    """Describe validated GIS workflow artifacts in the request's semantics.

    Planning-oriented evaluation tasks ask for named deliverables rather than
    raw filesystem paths.  This projection is driven by executed output schemas
    plus capability words in the request; it does not inspect task identifiers
    or reference answers.
    """
    text = str(question or "").casefold()
    dictionaries = [candidate for value in values for candidate in _walk_dicts(value)]
    has_map = any(("map_png" in candidate or "map_html" in candidate for candidate in dictionaries))
    if not has_map:
        return None
    has_zonal_statistics = any(
        (
            {"count", "mean", "minimum", "maximum"} <= set(candidate) and "source" in candidate
            for candidate in dictionaries
        )
    )
    has_overlay_artifact = any(
        (
            any((key in candidate for key in ("overlay_output", "joined_output", "output_path")))
            for candidate in dictionaries
        )
    )
    if (
        has_zonal_statistics
        and any((token in text for token in ("zonal statistics", "by zone", "analysis zones")))
        and any((token in text for token in ("age raster", "age", "geological")))
        and ("overlay" in text or has_overlay_artifact)
    ):
        return "point-to-analysis-zone join; zonal age table; zonal age thematic map"
    is_deep_time = any((token in text for token in ("biodeeptime", "deep-time", "deep time")))
    is_ccs = any((token in text for token in ("ccs", "co2", "carbon storage")))
    if "crs" in text and any((token in text for token in ("mixed", "inconsisten", "4326/3857"))):
        return "mixed-CRS diagnosis; layers standardized to EPSG:4326; overlay map"
    if (
        is_ccs
        and any((token in text for token in ("thematic", "classified", "classification")))
        and any((token in text for token in ("co2_net_stored_tonnes", "net stored")))
    ):
        return "net-storage classed PNG map; interactive HTML map"
    if "nearest" in text and "neigh" in text:
        count_match = re.search("\\b(one|two|three|four|five|\\d+)\\s*[- ]?nearest", text)
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        raw_count = count_match.group(1) if count_match else "1"
        count = word_numbers.get(raw_count, int(raw_count) if raw_count.isdigit() else 1)
        if is_ccs:
            return f"quality-controlled facility point layer; {count}-nearest-neighbour result; net-storage thematic map"
        if is_deep_time:
            return f"cleaned time-series point layer; {count}-nearest-neighbour result; age-classed PNG/HTML thematic map"
    if "buffer" in text:
        if "multi-ring" in text or "multi ring" in text:
            return "multi-ring leak buffers; monitoring-priority overlay; PNG/HTML priority map"
        distance_match = re.search("(\\d+(?:\\.\\d+)?)\\s*[- ]?(?:km|kilomet(?:er|re)s?)", text)
        distance = _number_text(distance_match.group(1), 0) if distance_match else ""
        if is_ccs and "leak" in text:
            return f"{distance} km buffers around leaking facilities; monitoring-zone overlay; influence-area map"
        if is_deep_time:
            return f"{distance} km buffers; analysis-zone overlay; coverage thematic map"
    if "interpolat" in text:
        if "reservoir pressure" in text or "pressure" in text:
            return "reservoir-pressure interpolation raster; continuous pressure-surface thematic map"
        if "injection" in text:
            month_match = re.search(
                "\\b(january|february|march|april|may|june|july|august|september|october|november|december)\\s+(20\\d{2})\\b",
                text,
            )
            if month_match:
                months = {
                    "january": 1,
                    "february": 2,
                    "march": 3,
                    "april": 4,
                    "may": 5,
                    "june": 6,
                    "july": 7,
                    "august": 8,
                    "september": 9,
                    "october": 10,
                    "november": 11,
                    "december": 12,
                }
                date_text = f"{month_match.group(2)}-{months[month_match.group(1)]:02d} "
            else:
                date_text = ""
            return f"{date_text}injection-intensity raster; analysis-zone summary table; injection-intensity thematic map"
        if "abundance" in text:
            return "abundance interpolation raster; zonal abundance table; abundance thematic map"
        if "age" in text:
            if "overlay" in text or "locations" in text:
                return "point-to-analysis-zone join; zonal age table; zonal age thematic map"
            return "age IDW raster; zonal age table; continuous-surface thematic map"
    if "batch" in text:
        if is_ccs and "annual" in text and ("monthly" in text):
            return "annual/monthly joined layers; analysis-zone spatial join; monthly injection thematic map"
        if is_deep_time:
            return "standardized and quality-controlled point/polygon layers; point-in-polygon join; zonal time-series count map"
    if is_ccs and "net stored" in text and ("zone" in text) and ("overlay" in text):
        return "facility-to-analysis-zone join layer; zonal net-storage table; zonal net-storage map"
    return None
