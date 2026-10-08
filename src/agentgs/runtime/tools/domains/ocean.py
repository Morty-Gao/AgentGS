"""ocean scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

from typing import Any

from agentgs.runtime.tools.context import paths, task_cache
from agentgs.runtime.tools.domains.shared import (
    _bbox_area,
    _haversine,
    _read_csv,
    _result,
    _row_dict,
)
from agentgs.runtime.tools.resources import _load_dataset_generator


def _match_row(table, filters: dict[str, Any]):
    rows = table
    for field, value in filters.items():
        if field not in rows.columns:
            continue
        if isinstance(value, (int, float)):
            rows = rows[(rows[field].astype(float) - float(value)).abs() < 1e-08]
        else:
            rows = rows[rows[field].astype(str).str.startswith(str(value))]
    if rows.empty:
        raise ValueError(f"No row matched filters {filters}")
    return rows.iloc[0]


def _ocean_grid(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "socat_monthly_grid_samples.csv")
    row = _match_row(
        table,
        {
            "date": params.get("date") or params.get("socat_date"),
            "latitude": params.get("latitude") or params.get("lat"),
            "longitude": params.get("longitude") or params.get("lon"),
        },
    )
    return _result(grid_cell=_row_dict(row))


def _ocean_profile(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "glodap_surface_deep_profiles.csv")
    row = _match_row(table, {"expocode": params["expocode"], "station": params["station"]})
    return _result(profile=_row_dict(row))


def _ocean_qc(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "snapo_cruise_qc_summary.csv")
    row = _match_row(table, {"cruise": params["cruise"]})
    return _result(qc_summary=_row_dict(row))


@task_cache(maxsize=1)
def _ocean_snapo_samples():
    generator = _load_dataset_generator("generate_ocean_carbon_toolqa")
    source_path = paths().root / "data" / "snapo" / "113597.xlsx"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"SNAPO source required for station distance is unavailable: {source_path}"
        )
    _, _, samples = generator.process_snapo(source_path)
    return samples


def _ocean_station_distance(params: dict[str, Any]) -> str:
    if params.get("snapo_cruise") is not None:
        profiles = _read_csv(
            paths().dataset("ocean") / "derived" / "glodap_surface_deep_profiles.csv"
        )
        first = _match_row(
            profiles,
            {
                "expocode": params.get("glodap_expocode") or params.get("expocode"),
                "station": params.get("glodap_station") or params.get("station"),
            },
        )
        second = _match_row(_ocean_snapo_samples(), {"cruise": params.get("snapo_cruise")})
        distance = _haversine(
            float(first["latitude"]),
            float(first["longitude"]),
            float(second["latitude"]),
            float(second["longitude"]),
        )
        return _result(
            source="GLODAP profile + selected SNAPO cruise observation",
            match={
                "glodap_expocode": str(first["expocode"]),
                "glodap_station": int(first["station"]),
                "glodap_latitude": float(first["latitude"]),
                "glodap_longitude": float(first["longitude"]),
                "snapo_cruise": str(second["cruise"]),
                "snapo_latitude": float(second["latitude"]),
                "snapo_longitude": float(second["longitude"]),
                "distance_km": distance,
            },
        )
    table = _read_csv(paths().dataset("ocean") / "derived" / "socat_glodap_nearest_samples.csv")
    filters = {
        "glodap_expocode": params.get("glodap_expocode"),
        "glodap_station": params.get("glodap_station"),
    }
    if params.get("socat_date") is not None:
        filters["socat_date"] = params.get("socat_date")
    row = _match_row(table, filters)
    return _result(source="GLODAP + SOCAT nearest-station table", match=_row_dict(row))


def _ocean_bbox(params: dict[str, Any]) -> str:
    return _result(bbox=params["bbox"], area_km2=_bbox_area(params["bbox"]))


def _ocean_annual(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "socat_north_pacific_year_summary.csv")
    return _result(summary=_row_dict(_match_row(table, {"year": params["year"]})))


def _ocean_vertical(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "glodap_surface_deep_profiles.csv")
    return _result(
        profile=_row_dict(
            _match_row(table, {"expocode": params["expocode"], "station": params["station"]})
        )
    )


def _ocean_map(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "socat_region_map_samples.csv")
    filters = {"region": params["region"]}
    if "year" in params:
        filters["year"] = params["year"]
    return _result(map_summary=_row_dict(_match_row(table, filters)))


def _ocean_match(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "socat_glodap_nearest_samples.csv")
    filters = {
        "glodap_expocode": params["glodap_expocode"],
        "glodap_station": params["glodap_station"],
    }
    if "socat_date" in params:
        filters["socat_date"] = params["socat_date"]
    return _result(match=_row_dict(_match_row(table, filters)))


def _ocean_dedup(params: dict[str, Any]) -> str:
    table = _read_csv(paths().dataset("ocean") / "derived" / "snapo_cruise_dedup_summary.csv")
    return _result(deduplication=_row_dict(_match_row(table, {"cruise": params["cruise"]})))
