"""hazard scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import json
from typing import Any

from agentgs.runtime.tools.context import paths, task_cache
from agentgs.runtime.tools.domains.shared import _clean, _result, _row_dict
from agentgs.runtime.tools.resources import _resolve_path


@task_cache(maxsize=1)
def _hazard_table():
    import pandas as pd

    return pd.read_excel(
        paths().root / "data/geotooleval/10.自然灾害与极端地震.xlsx", sheet_name="Sheet1"
    )


def _hazard_filtered(params: dict[str, Any]):
    import pandas as pd

    filtered_path = params.get("filtered_records_path")
    table = (
        pd.read_json(_resolve_path(filtered_path), lines=True)
        if filtered_path
        else _hazard_table().copy()
    )
    filters = params.get("filters") or {}
    if isinstance(filters, list):
        merged: dict[str, Any] = {}
        for item in filters:
            if isinstance(item, dict):
                merged.update(item)
        filters = merged
    if not isinstance(filters, dict):
        raise ValueError("Natural-hazard filters must be a JSON object")
    for field, value in filters.items():
        if field not in table.columns:
            raise ValueError(f"Unknown natural-hazard field: {field}")
        if isinstance(value, list):
            table = table[table[field].isin(value)]
        elif value is None:
            table = table[table[field].isna()]
        else:
            table = table[table[field] == value]
    return table


def _hazard_load(params: dict[str, Any]) -> str:
    table = _hazard_table()
    return _result(
        source=paths().root / "data/geotooleval/10.自然灾害与极端地震.xlsx",
        sheet="Sheet1",
        record_count=int(len(table)),
        columns=list(table.columns),
    )


def _hazard_filter(params: dict[str, Any]) -> str:
    filters = params.get("filters")
    if isinstance(filters, list):
        filters = {
            key: value for item in filters if isinstance(item, dict) for key, value in item.items()
        }
        params = {**params, "filters": filters}
    if not isinstance(filters, dict) or not filters:
        raise ValueError("NaturalHazardFilterRecords requires a non-empty filters object")
    table = _hazard_filtered(params)
    digest = abs(hash(json.dumps(filters, ensure_ascii=False, sort_keys=True)))
    output = paths().outputs / "natural_hazard" / f"filtered_{digest}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_json(output, orient="records", lines=True, force_ascii=False, date_format="iso")
    return _result(filters=filters, record_count=int(len(table)), filtered_records_path=output)


def _hazard_count(params: dict[str, Any]) -> str:
    return _result(filters=params.get("filters", {}), count=int(len(_hazard_filtered(params))))


def _hazard_sum(params: dict[str, Any]) -> str:
    table = _hazard_filtered(params)
    field = str(params.get("measure") or params.get("field"))
    return _result(
        filters=params.get("filters", {}), measure=field, sum=float(table[field].fillna(0).sum())
    )


def _hazard_group(params: dict[str, Any]) -> str:
    table = _hazard_filtered(params)
    group_by = str(params["group_by"])
    measure = params.get("measure")
    if measure:
        ranked = (
            table.groupby(group_by, dropna=False)[str(measure)]
            .sum(min_count=1)
            .sort_values(ascending=False)
        )
        aggregate = "sum"
    else:
        ranked = table.groupby(group_by, dropna=False).size().sort_values(ascending=False)
        aggregate = "count"
    records = [
        {group_by: _clean(index), aggregate: _clean(value)} for index, value in ranked.items()
    ]
    return _result(
        filters=params.get("filters", {}),
        group_by=group_by,
        aggregate=aggregate,
        ranking=records,
        top=records[0] if records else None,
    )


def _hazard_max(params: dict[str, Any]) -> str:
    table = _hazard_filtered(params)
    field = str(params.get("measure") or params.get("field"))
    row = table.loc[table[field].astype(float).idxmax()]
    return _result(filters=params.get("filters", {}), measure=field, event=_row_dict(row))


def _hazard_dates(params: dict[str, Any]) -> str:
    import pandas as pd

    table = _hazard_filtered(params)
    fields = params.get("date_fields") or ["Start Year", "Start Month", "Start Day"]
    dates = pd.to_datetime(
        dict(year=table[fields[0]], month=table[fields[1]], day=table[fields[2]]), errors="coerce"
    )
    valid = dates.dropna()
    return _result(
        filters=params.get("filters", {}),
        valid_date_count=int(valid.size),
        latest_date=None if valid.empty else valid.max().date().isoformat(),
    )


def _hazard_latest(params: dict[str, Any]) -> str:
    import pandas as pd

    table = _hazard_filtered(params).copy()
    fields = params.get("date_fields") or ["Start Year", "Start Month", "Start Day"]
    table["_start_date"] = pd.to_datetime(
        dict(year=table[fields[0]], month=table[fields[1]], day=table[fields[2]]), errors="coerce"
    )
    row = table.dropna(subset=["_start_date"]).sort_values("_start_date").iloc[-1]
    return _result(
        filters=params.get("filters", {}),
        latest_date=row["_start_date"].date().isoformat(),
        event=_row_dict(row.drop(labels=["_start_date"])),
    )
