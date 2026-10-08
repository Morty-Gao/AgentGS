"""Explicit source roles and observation selection for preserved spatial tools.

This module does not inspect questions, expected answers, filenames as semantics,
or benchmark identifiers. Ambiguous sources and missing selection fields fail.
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def explicit_spatial_paths(params, operation, resolve, fallback):
    if operation == "idw" and (params.get("points_path") or params.get("point_path")):
        return [resolve(params.get("points_path") or params["point_path"])]
    if operation in {"raster", "render", "zonal"} and params.get("raster_path"):
        paths = [resolve(params["raster_path"])]
        if operation == "zonal":
            if not params.get("zones_path"):
                raise ValueError("Zonal statistics with raster_path require an explicit zones_path")
            paths.append(resolve(params["zones_path"]))
        return paths
    if operation == "render" and params.get("vector_path"):
        return [resolve(params["vector_path"])]
    if operation == "zonal" and params.get("table_path"):
        return [resolve(params["table_path"])]
    return list(dict.fromkeys(fallback(params)))


def _month(value):
    text = str(value).strip()
    if re.fullmatch("\\d{4}-\\d{2}", text):
        datetime.strptime(text, "%Y-%m")
        return text
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed.strftime("%Y-%m")


def select_observation_rows(params, paths, output_root):
    filters = params.get("filters") or {}
    selected_month = params.get("month")
    date_field = params.get("date_field")
    if not isinstance(filters, dict):
        raise ValueError(
            "filters must be a field-to-exact-value object; a list value explicitly selects one of those values"
        )
    if "allow_temporal_mixing" in params and type(params["allow_temporal_mixing"]) is not bool:
        raise ValueError("allow_temporal_mixing must be an explicit JSON boolean")
    unsupported = [
        key
        for key in ("date", "year", "start_date", "end_date", "date_range", "time_range")
        if params.get(key) is not None
    ]
    if unsupported:
        raise ValueError(
            "Unsupported temporal selectors: "
            + ", ".join(unsupported)
            + "; use explicit date_field and month=YYYY-MM"
        )
    if selected_month is not None:
        if not re.fullmatch("\\d{4}-\\d{2}", str(selected_month)):
            raise ValueError("month must use YYYY-MM")
        selected_month = _month(selected_month)
        if not isinstance(date_field, str) or not date_field:
            raise ValueError("month selection requires an explicit date_field")
    if len(paths) != 1 or paths[0].suffix.casefold() != ".csv":
        if filters or selected_month is not None:
            raise ValueError("Observation filtering requires exactly one CSV; provide points_path")
        return (paths, {})
    source = paths[0]
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError("No input point observations")
    for field, expected in filters.items():
        if field not in fields or isinstance(expected, dict):
            raise ValueError("Unknown filter field or unsupported filter operator: " + str(field))
        choices = expected if isinstance(expected, list) else [expected]
        if not choices or any((isinstance(value, (dict, list)) for value in choices)):
            raise ValueError("Filter choices must be nonempty scalar values")
    if date_field is not None and date_field not in fields:
        raise ValueError("date_field is not present in the point CSV")
    temporal_fields = (
        [date_field]
        if date_field
        else [field for field in ("month", "date", "timestamp", "time") if field in fields]
    )
    if not filters and selected_month is None and (not params.get("allow_temporal_mixing", False)):
        for field in temporal_fields:
            values = {row[field] for row in rows}
            if len(values) > 1:
                raise ValueError(
                    "Point observations span multiple "
                    + field
                    + " values; select date_field+month or explicit filters before IDW, or explicitly allow_temporal_mixing"
                )
    if not filters and selected_month is None:
        return (
            paths,
            {
                "source": str(source),
                "selected_rows": len(rows),
                "source_rows": len(rows),
                "temporal_mixing_explicitly_allowed": bool(
                    params.get("allow_temporal_mixing", False)
                ),
            },
        )
    kept = []
    for row in rows:
        keep = all(
            (
                str(row[field])
                in {
                    str(value) for value in (expected if isinstance(expected, list) else [expected])
                }
                for field, expected in filters.items()
            )
        )
        if selected_month is not None:
            keep = keep and _month(row[date_field]) == selected_month
        if keep:
            kept.append(row)
    if not kept:
        raise ValueError("No observations match the explicit filters/month")
    if selected_month is None and (not params.get("allow_temporal_mixing", False)):
        for field in temporal_fields:
            if len({row[field] for row in kept}) > 1:
                raise ValueError(
                    "Selected observations still span multiple "
                    + field
                    + " values; add an explicit temporal selection"
                )
    output = Path(output_root) / "selected_observations" / ("points_" + uuid4().hex + ".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    return (
        [output],
        {
            "source": str(source),
            "selected_source": str(output),
            "source_rows": len(rows),
            "selected_rows": len(kept),
            "filters": filters,
            "date_field": date_field,
            "month": selected_month,
        },
    )
