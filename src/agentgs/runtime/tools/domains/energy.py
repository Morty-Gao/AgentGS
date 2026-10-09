"""energy scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import math
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


def _energy_facility_distance(params: dict[str, Any]) -> str:
    municipality = params.get("solar_municipality")
    table = _read_csv(paths().data("energy", "derived/opsd_landmark_facilities.csv"))
    country = str(params["country"])
    wind_municipality = params.get("wind_municipality")
    missing_solar_locator = (
        municipality is None
        or (isinstance(municipality, float) and math.isnan(municipality))
        or str(municipality).strip().casefold() in {"", "nan", "none", "null"}
    )
    if missing_solar_locator:
        solar_rows = table[
            (table["country"] == country)
            & (table["energy_source"] == "Solar")
            & table["municipality"].isna()
        ]
        wind_rows = table[
            (table["country"] == country)
            & (table["energy_source"] == "Wind")
            & (table["municipality"].astype(str) == str(wind_municipality))
        ]
        if solar_rows.empty or wind_rows.empty:
            raise ValueError("No facilities match the available country/municipality constraints")
        candidates = []
        for _, solar_row in solar_rows.iterrows():
            for _, wind_row in wind_rows.iterrows():
                candidates.append(
                    {
                        "distance_km": _haversine(
                            float(solar_row["latitude"]),
                            float(solar_row["longitude"]),
                            float(wind_row["latitude"]),
                            float(wind_row["longitude"]),
                        ),
                        "solar": _row_dict(solar_row),
                        "wind": _row_dict(wind_row),
                    }
                )
        return _result(
            ambiguous_locator=True,
            candidate_count=len(candidates),
            candidates=candidates,
            distances_km=[item["distance_km"] for item in candidates],
        )
    solar = table[
        (table["country"] == country)
        & (table["municipality"].astype(str) == str(params["solar_municipality"]))
        & (table["energy_source"] == "Solar")
    ].iloc[0]
    wind = table[
        (table["country"] == country)
        & (table["municipality"].astype(str) == str(wind_municipality))
        & (table["energy_source"] == "Wind")
    ].iloc[0]
    distance = _haversine(
        float(solar["latitude"]),
        float(solar["longitude"]),
        float(wind["latitude"]),
        float(wind["longitude"]),
    )
    return _result(distance_km=distance, solar=_row_dict(solar), wind=_row_dict(wind))


def _energy_mineral_distance(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("energy", "derived/mrds_critical_mineral_snapshot.csv"))
    first = table[table["dep_id"].astype(int) == int(params["dep_id_1"])].iloc[0]
    second = table[table["dep_id"].astype(int) == int(params["dep_id_2"])].iloc[0]
    distance = _haversine(
        float(first["latitude"]),
        float(first["longitude"]),
        float(second["latitude"]),
        float(second["longitude"]),
    )
    return _result(distance_km=distance, first=_row_dict(first), second=_row_dict(second))


def _energy_bbox(params: dict[str, Any]) -> str:
    bbox = params["bbox"]
    return _result(bbox=bbox, area_km2=_bbox_area(bbox))


def _energy_nearest(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("energy", "derived/opsd_capital_nearest.csv"))
    rows = table[
        (table["country"] == params["country"])
        & (table["energy_source"] == params["energy_source"])
    ]
    return _result(nearest=_row_dict(rows.sort_values("distance_km").iloc[0]))


def _capacity_row(series: str, year: int):
    import pandas as pd

    table = _read_csv(paths().data("energy", "derived/opsd_capacity_year_end.csv"))
    table["day"] = pd.to_datetime(table["day"])
    rows = table[table["day"].dt.year == year]
    if rows.empty:
        raise ValueError(f"No capacity sample for year {year}")
    return (rows.sort_values("day").iloc[-1], table)


def _energy_capacity(params: dict[str, Any]) -> str:
    series = str(params.get("series") or params.get("capacity_series") or "")
    year_value = params.get("year") or params.get("horizon_year")
    if isinstance(year_value, (list, tuple)):
        year_value = next((item for item in reversed(year_value) if str(item).isdigit()), None)
    if year_value is None:
        year_value = 2020
    row, _ = _capacity_row(series, int(year_value))
    return _result(series=series, year=int(year_value), capacity_mw=float(row[series]))


def _energy_cross_nearest(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("energy", "derived/power_mineral_nearest_samples.csv"))
    rows = table[table["facility"].astype(str) == str(params["plant_name"])]
    return _result(match=_row_dict(rows.iloc[0]))


def _energy_mix(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("energy", "derived/gip_operating_country_type.csv"))
    rows = table[table["country"].astype(str) == str(params["country"])]
    if params.get("renewable_types"):
        rows = rows[rows["type"].isin(params["renewable_types"])]
    largest = rows.sort_values("capacity_mw", ascending=False).iloc[0]
    return _result(
        total_capacity_mw=float(rows["capacity_mw"].sum()),
        largest_type=_row_dict(largest),
        by_type=[_row_dict(row) for _, row in rows.iterrows()],
    )


def _energy_growth(params: dict[str, Any]) -> str:
    series = str(params["series"])
    start, _ = _capacity_row(series, int(params["start_year"]))
    end, _ = _capacity_row(series, int(params["end_year"]))
    a, b = (float(start[series]), float(end[series]))
    return _result(
        start_mw=a,
        end_mw=b,
        absolute_growth_mw=b - a,
        percentage_growth=None if a == 0 else (b - a) / a * 100,
    )


def _energy_transition(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("energy", "derived/country_transition_metrics.csv"))
    rows = table[table["country"].astype(str) == str(params["country"])]
    return _result(metrics=_row_dict(rows.iloc[0]))


@task_cache(maxsize=1)
def _energy_full_overlay_sources():
    """Load full normalized GEM/MRDS point sources for spatial overlays.

    The small ``top`` and ``snapshot`` derivatives are suitable for lookup
    examples, not bbox population statistics.  Parse the retained authoritative
    sources once and cache their normalized point tables for subsequent calls.
    """
    facilities = paths().data("energy", "full/gip_renewable_points.csv.gz")
    minerals = paths().data("energy", "full/mrds_critical_mineral_points.csv.gz")
    if facilities.is_file() and minerals.is_file():
        return (_read_csv(facilities), _read_csv(minerals))
    generator = _load_dataset_generator("generate_energy_resources_toolqa")
    extracted = paths().root / "data/raw"
    gip_path = paths().root / "data/gem.jsonl"
    mrds_path = extracted / "mrds" / "mrds.csv"
    if not gip_path.is_file() or not mrds_path.is_file():
        raise FileNotFoundError(
            "Full GEM/MRDS sources required for regional overlay are unavailable"
        )
    _, _, facilities = generator.process_gip(gip_path)
    minerals = generator.process_mrds(mrds_path)
    return (facilities, minerals)


def _energy_overlay(params: dict[str, Any]) -> str:
    region = str(params.get("region") or "")
    bbox = params.get("bbox")
    facilities, minerals = _energy_full_overlay_sources()
    facilities = facilities.copy()
    minerals = minerals.copy()
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = [float(value) for value in bbox]
        facilities = facilities[
            (facilities["longitude"].astype(float) >= min_lon)
            & (facilities["longitude"].astype(float) <= max_lon)
            & (facilities["latitude"].astype(float) >= min_lat)
            & (facilities["latitude"].astype(float) <= max_lat)
        ]
        minerals = minerals[
            (minerals["longitude"].astype(float) >= min_lon)
            & (minerals["longitude"].astype(float) <= max_lon)
            & (minerals["latitude"].astype(float) >= min_lat)
            & (minerals["latitude"].astype(float) <= max_lat)
        ]
    return _result(
        source="GEM operating renewable facilities + full MRDS critical-mineral set",
        region=region,
        facility_count=int(len(facilities)),
        mineral_site_count=int(len(minerals)),
        total_capacity_mw=float(facilities["capacity_mw"].astype(float).sum())
        if "capacity_mw" in facilities
        else 0.0,
    )
