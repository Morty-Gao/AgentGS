"""mineral scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from agentgs.runtime.tools.context import paths
from agentgs.runtime.tools.domains.shared import _read_csv, _result, _row_dict


def _mineral_table():
    return _read_csv(paths().data("habitable", "metal_minerals_subset.csv"))


def _mineral_search(params: dict[str, Any]) -> str:
    table = _mineral_table()
    country = str(params.get("country", ""))
    commodity = str(params.get("commodity") or params.get("commod1") or "")
    limit = int(params.get("limit", 3))
    rows = table
    if country:
        rows = rows[rows["country"].astype(str).str.casefold() == country.casefold()]
    if commodity:
        rows = rows[
            rows["commod1"]
            .astype(str)
            .str.casefold()
            .str.contains(re.escape(commodity.casefold()), na=False)
        ]
    return _result(
        count=int(len(rows)), deposits=[_row_dict(row) for _, row in rows.head(limit).iterrows()]
    )


def _mineral_get(params: dict[str, Any]) -> str:
    dep_id = int(params.get("dep_id") or params.get("value"))
    table = _mineral_table()
    rows = table[table["dep_id"].astype(int) == dep_id]
    if rows.empty:
        raise ValueError(f"Unknown MRDS dep_id: {dep_id}")
    return _result(deposit=_row_dict(rows.iloc[0]))


def _mineral_country_summary(params: dict[str, Any]) -> str:
    country = str(params.get("country") or params.get("input_text") or "")
    limit = int(params.get("limit", 3))
    table = _mineral_table()
    rows = table[table["country"].astype(str).str.casefold() == country.casefold()]
    counts = rows["commod1"].fillna("Unknown").astype(str).value_counts()
    constituent_support: Counter[str] = Counter()
    for label in table["commod1"].dropna().astype(str):
        constituent_support.update(
            (token.strip().casefold() for token in label.split(",") if token.strip())
        )

    def _support(label: str) -> int:
        return sum(
            (
                constituent_support[token.strip().casefold()]
                for token in label.split(",")
                if token.strip()
            )
        )

    ranked = sorted(
        ((str(name), int(count)) for name, count in counts.items()),
        key=lambda item: (
            -item[1],
            item[0].casefold() == "unknown",
            -_support(item[0]),
            item[0].casefold(),
        ),
    )[:limit]
    return _result(country=country, record_count=int(len(rows)), top_commodities=dict(ranked))


def _contains_exact_commodity_token(value: Any, commodity: str) -> bool:
    """Return whether a comma-delimited commodity field contains the target.

    The broad row filter intentionally keeps substring matching for backward
    compatibility (for example, catalog variants and compound labels).  Exact
    token matches are used only as a stable relevance tie-break at the Top-K
    boundary, so equally frequent countries with stronger primary evidence are
    preferred without changing their aggregate counts.
    """
    target = commodity.strip().casefold()
    if not target or value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return any((token.strip().casefold() == target for token in str(value).split(",")))


def _mineral_rank_commodity(params: dict[str, Any]) -> str:
    commodity = str(params.get("commodity") or params.get("input_text") or "")
    limit = int(params.get("limit", 5))
    table = _mineral_table()
    fields = table[["commod1", "commod2"]].fillna("").astype(str).agg(", ".join, axis=1)
    rows = table[fields.str.casefold().str.contains(re.escape(commodity.casefold()), na=False)]
    country_labels = rows["country"].fillna("Unknown").astype(str)
    counts = country_labels.value_counts()
    primary_exact: dict[str, int] = {}
    secondary_exact: dict[str, int] = {}
    for row_index, country_label in country_labels.items():
        primary_exact[country_label] = primary_exact.get(country_label, 0) + int(
            _contains_exact_commodity_token(rows.at[row_index, "commod1"], commodity)
        )
        secondary_exact[country_label] = secondary_exact.get(country_label, 0) + int(
            _contains_exact_commodity_token(rows.at[row_index, "commod2"], commodity)
        )
    ranked = sorted(
        ((str(name), int(count)) for name, count in counts.items()),
        key=lambda item: (
            -item[1],
            -primary_exact.get(item[0], 0),
            -secondary_exact.get(item[0], 0),
            item[0].casefold() == "unknown",
            item[0].casefold(),
        ),
    )[:limit]
    return _result(commodity=commodity, countries=dict(ranked))
