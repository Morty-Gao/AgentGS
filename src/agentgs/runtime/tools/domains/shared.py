"""shared scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from agentgs.contracts.arguments import parse_arguments
from agentgs.runtime.tools.resources import _resolve_path

_params = parse_arguments


def _first_number(value: Any) -> float | None:
    """Return the first finite numeric leaf from an upstream result object."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        preferred = (
            "value",
            "mid",
            "mean",
            "numeric_result",
            "value_km",
            "temp_anomaly_c",
            "dlnvs_percent",
        )
        for key in preferred:
            if key in value:
                number = _first_number(value[key])
                if number is not None:
                    return number
        for item in value.values():
            number = _first_number(item)
            if number is not None:
                return number
    if isinstance(value, (list, tuple)):
        for item in value:
            number = _first_number(item)
            if number is not None:
                return number
    return None



def _clean(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat") and type(value).__module__.startswith(("pandas", "numpy")):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _result(**payload: Any) -> str:
    return json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True)


def _read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _row_dict(row) -> dict[str, Any]:
    return {str(key): _clean(value) for key, value in row.to_dict().items()}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = (math.radians(lat1), math.radians(lat2))
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bbox_area(bbox: list[float]) -> float:
    min_lon, min_lat, max_lon, max_lat = [float(value) for value in bbox]
    radius_km = 6371.0088
    return abs(
        radius_km**2
        * math.radians(max_lon - min_lon)
        * (math.sin(math.radians(max_lat)) - math.sin(math.radians(min_lat)))
    )


def _point_pair(params: dict[str, Any]) -> tuple[float, float, float, float]:
    text = str(params.get("input_text", ""))
    if text:
        left, right = text.split(";", 1)
        lat1, lon1 = [float(value.strip()) for value in left.split(",")]
        lat2, lon2 = [float(value.strip()) for value in right.split(",")]
        return (lat1, lon1, lat2, lon2)
    if all((params.get(key) is not None for key in ("lat1", "lon1", "lat2", "lon2"))):
        return tuple((float(params[key]) for key in ("lat1", "lon1", "lat2", "lon2")))
    identifiers = params.get("series_ids") or params.get("case_ids") or params.get("facility_ids")
    if isinstance(identifiers, list) and len(identifiers) >= 2:
        if params.get("series_ids"):
            path = _resolve_path("biodeeptime_geospatial_subset.csv")
            table = _read_csv(path).drop_duplicates("seriesID")
            id_field, lon_field = ("seriesID", "long")
        else:
            path = _resolve_path("ccs_full_dataset_v1.0.csv")
            table = _read_csv(path).drop_duplicates("case_id")
            id_field, lon_field = ("case_id", "lon")
        points = []
        for identifier in identifiers[:2]:
            rows = table[table[id_field].astype(str) == str(identifier)]
            if rows.empty:
                raise ValueError(f"Unknown spatial identifier: {identifier}")
            row = rows.iloc[0]
            points.extend((float(row["lat"]), float(row[lon_field])))
        return tuple(points)
    raise ValueError(
        "A coordinate pair requires lat1/lon1/lat2/lon2, compact input_text, series_ids, or case_ids"
    )


def _numeric_input(params: dict[str, Any]) -> str:
    if params.get("input_text") not in (None, ""):
        return str(params["input_text"])
    value = params.get("value")
    axis = params.get("field") or params.get("axis")
    if value is None and (params.get("case_id") or params.get("facility_id")):
        table = _read_csv(_resolve_path("ccs_full_dataset_v1.0.csv"))
        id_field = "case_id" if params.get("case_id") else "facility_id"
        identifier = params.get(id_field)
        rows = table[table[id_field].astype(str) == str(identifier)]
        if not rows.empty and axis in rows.columns:
            value = rows.iloc[0][str(axis)]
    if value is None and params.get("series_id"):
        table = _read_csv(_resolve_path("biodeeptime_geospatial_subset.csv"))
        rows = table[table["seriesID"].astype(str) == str(params["series_id"])]
        field = "long" if axis == "lon" else axis
        if not rows.empty and field in rows.columns:
            value = rows.iloc[0][str(field)]
    return (
        f"{value},{axis}" if value is not None and axis else str(value if value is not None else "")
    )
