"""deep earth scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

from typing import Any

from agentgs.runtime.tools.context import paths, task_cache
from agentgs.runtime.tools.domains.shared import _read_csv, _result, _row_dict


def _deep_litho(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("deep_earth", "derived/location_model_samples.csv"))
    key = params.get("location_key") or params.get("key")
    rows = table[table["key"].astype(str) == str(key)]
    if rows.empty:
        raise ValueError(f"Unknown LITHO1.0 location_key: {key}")
    row = rows.iloc[0]
    layer = str(params.get("layer") or params.get("field") or "moho").lower()
    field = "lab_depth_km" if any((token in layer for token in ("lid", "lab"))) else "moho_depth_km"
    return _result(
        source="LITHO1.0 pinned nearest-node sample",
        location_key=key,
        node_index=int(row["node_index"]),
        sampled_latitude=float(row["sampled_latitude"]),
        sampled_longitude=float(row["sampled_longitude"]),
        distance_to_node_km=float(row["distance_to_node_km"]),
        field=field,
        value_km=float(row[field]),
    )


def _deep_semucb(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("deep_earth", "derived/location_model_samples.csv"))
    key = params.get("location_key") or params.get("key")
    depth = int(params.get("depth_km", 100))
    field = f"semucb_dlnvs_{depth}km_percent"
    if field not in table.columns:
        raise ValueError(f"SEMUCB-WM1 pinned samples do not include depth {depth} km")
    rows = table[table["key"].astype(str) == str(key)]
    if rows.empty:
        raise ValueError(f"Unknown SEMUCB-WM1 location_key: {key}")
    row = rows.iloc[0]
    return _result(
        source="SEMUCB-WM1 pinned spline sample",
        location_key=key,
        depth_km=depth,
        radius_km=6371.0 - depth,
        dlnvs_percent=float(row[field]),
    )


@task_cache(maxsize=1)
def _usgs_catalog():
    import pandas as pd

    path = paths().data("deep_earth", "source_snapshot/usgs_2010_2020_m7.csv")
    if not path.is_file():
        path = paths().root / "data/raw/usgs/usgs_2010_2020_m7.csv"
    table = pd.read_csv(path)
    table["time"] = pd.to_datetime(table["time"], utc=True)
    return table


def _deep_usgs_event(params: dict[str, Any]) -> str:
    event_id = str(params.get("event_id") or params.get("id"))
    table = _usgs_catalog()
    rows = table[table["id"].astype(str) == event_id]
    if rows.empty:
        raise ValueError(f"Unknown USGS event_id: {event_id}")
    return _result(source="USGS frozen 2010-2020 M7+ catalog", event=_row_dict(rows.iloc[0]))


def _deep_usgs_query(params: dict[str, Any]) -> str:
    import pandas as pd

    table = _usgs_catalog().copy()
    operation = str(params.get("operation", "")).lower()
    if "minimum_magnitude" in params:
        table = table[table["mag"] >= float(params["minimum_magnitude"])]
    if "year" in params:
        table = table[table["time"].dt.year == int(params["year"])]
    if "start" in params:
        table = table[table["time"] >= pd.Timestamp(params["start"], tz="UTC")]
    if "end_exclusive" in params:
        table = table[table["time"] < pd.Timestamp(params["end_exclusive"], tz="UTC")]
    if params.get("event_id") and operation in ("", "event", "lookup"):
        return _deep_usgs_event(params)
    if operation in ("largest", "max_magnitude", "rank_magnitude"):
        row = table.sort_values(["mag", "time"], ascending=[False, True]).iloc[0]
        return _result(operation="max_magnitude", event=_row_dict(row))
    if operation in ("deepest", "max_depth"):
        row = table.sort_values(["depth", "time"], ascending=[False, True]).iloc[0]
        return _result(operation="deepest", event=_row_dict(row))
    if operation in ("total_energy", "sum_energy"):
        energies = 10 ** (1.5 * table["mag"].astype(float) + 4.8)
        return _result(
            operation="sum_energy", event_count=len(table), total_energy_j=float(energies.sum())
        )
    return _result(
        operation="count", count=int(len(table)), event_ids=table["id"].astype(str).tolist()
    )


def _deep_event_models(params: dict[str, Any]) -> str:
    event_id = str(params.get("event_id") or params.get("id"))
    table = _read_csv(paths().data("deep_earth", "derived/cross_event_samples.csv"))
    rows = table[table["event_id"].astype(str) == event_id]
    if rows.empty:
        raise ValueError(f"No pinned three-model sample for event_id: {event_id}")
    row = rows.iloc[0]
    return _result(source="USGS + LITHO1.0 + SEMUCB-WM1 pinned join", sample=_row_dict(row))
