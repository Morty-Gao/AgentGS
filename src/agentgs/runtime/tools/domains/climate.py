"""climate scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

from typing import Any

from agentgs.runtime.tools.context import paths
from agentgs.runtime.tools.domains.shared import _first_number, _read_csv, _result, _row_dict


def _climate_deep(params: dict[str, Any]) -> str:
    dataset = str(params.get("dataset", "friedrich2016")).lower()
    filename = (
        "deep_time_bintanja2005_temp_sealevel.csv"
        if "bintanja" in dataset
        else "deep_time_friedrich2016_temp.csv"
    )
    table = _read_csv(paths().data("habitable", filename))
    if str(params.get("find", "")).lower() in ("min_temp", "min_temp_friedrich", "coldest"):
        row = table.loc[table["temp_anomaly_C"].astype(float).idxmin()]
        return _result(dataset=dataset, nearest=_row_dict(row))
    ages = params.get("ages_kyr_bp")
    if isinstance(ages, (list, tuple)) and ages:
        requested = [float(age) for age in ages]
        nearest = [
            _row_dict(table.iloc[(table["age_kyr_bp"].astype(float) - age).abs().argmin()])
            for age in requested
        ]
        return _result(dataset=dataset, requested_ages_kyr_bp=requested, nearest=nearest)
    age = float(params.get("age_kyr_bp", 0))
    row = table.iloc[(table["age_kyr_bp"].astype(float) - age).abs().argmin()]
    return _result(dataset=dataset, nearest=_row_dict(row))


def _climate_present(params: dict[str, Any]) -> str:
    table = _read_csv(paths().data("habitable", "present_climate_gistemp_global_annual.csv"))
    start = int(params.get("year_start", table["year"].min()))
    end = int(params.get("year_end", params.get("modern_year", table["year"].max())))
    if end < start:
        start, end = (end, start)
    rows = table[(table["year"] >= start) & (table["year"] <= end)]
    return _result(
        year_start=start,
        year_end=end,
        n=int(len(rows)),
        mean_temp_anomaly_c=float(rows["temp_anomaly_C"].mean()),
    )


def _climate_ssp(params: dict[str, Any]) -> str:
    scenario = str(params.get("scenario") or params.get("ssp") or "")
    year_value = (
        params.get("horizon_year") or params.get("ssp_horizon") or params.get("year") or 2100
    )
    if isinstance(year_value, (list, tuple)):
        year_value = next((item for item in reversed(year_value) if str(item).isdigit()), 2100)
    year = int(year_value)
    table = _read_csv(paths().data("habitable", "future_climate_ssp_warming_summary.csv"))
    rows = table[(table["scenario"] == scenario) & (table["horizon_year"] == year)]
    if rows.empty:
        raise ValueError(f"Unknown SSP scenario/horizon: {scenario}/{year}")
    return _result(scenario=scenario, horizon_year=year, estimate=_row_dict(rows.iloc[0]))


def _climate_compare(params: dict[str, Any]) -> str:
    values = params.get("values") or params.get("x")
    if values is None:
        values = [params.get("a"), params.get("b")]
    numbers = []
    for value in values if isinstance(values, (list, tuple)) else [values]:
        number = _first_number(value)
        if number is not None:
            numbers.append(number)
    if len(numbers) < 2:
        raise ValueError("ClimateCompareEpochs requires at least two numeric values")
    baseline = numbers[0]
    return _result(values=numbers, differences_from_first=[value - baseline for value in numbers])
