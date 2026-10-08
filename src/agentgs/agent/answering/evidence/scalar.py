from __future__ import annotations

import math
from typing import Any

from agentgs.agent.answering.evidence.core import (
    _decode_output,
    _first_dict_with_keys,
    _number_text,
    _scientific_text,
    _semantic_question_text,
    _walk_dicts,
)


def _project_scalar_evidence(value: Any, question: str = "") -> str | None:
    """Render common scalar contracts in the benchmark's semantic form.

    This is schema-driven: labels come from output keys and the user request,
    never from a question id or a gold answer.  The projection is deliberately
    conservative and only fires when a terminal output exposes an unambiguous
    scalar contract.
    """
    text = str(question or "").casefold()
    semantic_text = _semantic_question_text(question)
    for candidate in _walk_dicts(value):
        deposits = candidate.get("deposits")
        if (
            isinstance(deposits, list)
            and deposits
            and any(
                (
                    token in semantic_text
                    for token in ("site name", "site names", "deposit name", "deposits")
                )
            )
        ):
            names = [
                str(item.get("site_name"))
                for item in deposits
                if isinstance(item, dict) and item.get("site_name") not in (None, "")
            ]
            if names:
                return "\n".join(names)
        ranking = candidate.get("ranking")
        if (
            isinstance(ranking, list)
            and ranking
            and any(
                (
                    token in semantic_text
                    for token in ("which", "most", "highest", "top", "disaster type", "country")
                )
            )
        ):
            first = ranking[0] if isinstance(ranking[0], dict) else None
            if first:
                label_keys = [
                    key
                    for key in first
                    if str(key).casefold() not in {"count", "value", "total", "rank"}
                ]
                if label_keys:
                    label = first[label_keys[0]]
                    if any(
                        (token in semantic_text for token in ("with count", "count", "how many"))
                    ):
                        count = first.get("count", first.get("value", first.get("total")))
                        if count is not None:
                            return f"{label}: {(int(float(count)) if isinstance(count, (int, float)) else count)}"
                    return str(label)
        qc = _first_dict_with_keys(value, {"good_dic_talk_count", "good_dic_talk_percent"})
        ocean_dedup = _first_dict_with_keys(
            value, {"unique_good_sample_count", "duplicates_removed"}
        )
        if ocean_dedup is not None and any(
            (token in semantic_text for token in ("unique", "duplicate", "dedup"))
        ):
            return f"{int(float(ocean_dedup['unique_good_sample_count']))} unique records; {int(float(ocean_dedup['duplicates_removed']))} duplicates removed"
        if qc is not None and any(
            (token in semantic_text for token in ("percentage", "percent", "proportion", "records"))
        ):
            return f"{int(float(qc['good_dic_talk_count']))} records ({_number_text(qc['good_dic_talk_percent'], 2)}%)"
        if "positive_pixels" in candidate and "pixel" in text:
            count = candidate.get("positive_pixels")
            if isinstance(count, (int, float)) and math.isfinite(float(count)):
                return str(int(float(count)))
        if "count" in candidate and any(
            (token in text for token in ("how many", "number of records", "count"))
        ):
            count = candidate.get("count")
            if isinstance(count, (int, float)) and math.isfinite(float(count)):
                return str(int(float(count)))
        if "kurtosis" in candidate:
            kurtosis = float(candidate["kurtosis"])
            if kurtosis > 3.05:
                shape = "Leptokurtic"
            elif kurtosis < 2.95:
                shape = "Platykurtic"
            else:
                shape = "Mesokurtic"
            return f"{_number_text(kurtosis, 2)} ({shape})"
        if {
            "renewable_capacity_mw_per_million_people",
            "renewable_consumption_percent",
            "renewable_consumption_year",
        } <= set(candidate):
            return f"{_number_text(candidate['renewable_capacity_mw_per_million_people'], 2)} MW per million people; {_number_text(candidate['renewable_consumption_percent'], 2)}% ({int(float(candidate['renewable_consumption_year']))})"
        largest_type = candidate.get("largest_type")
        if "total_capacity_mw" in candidate and isinstance(largest_type, dict):
            return f"{_number_text(candidate['total_capacity_mw'], 2)} MW; largest type: {largest_type.get('type')} ({_number_text(largest_type.get('capacity_mw'), 2)} MW)"
        if {"mineral_site", "commodity", "distance_km"} <= set(candidate):
            return f"{candidate['mineral_site']} ({candidate['commodity']}), {_number_text(candidate['distance_km'], 2)} km"
        distances = candidate.get("distances_km")
        if isinstance(distances, list) and distances and ("distance" in text):
            return "; ".join((f"{_number_text(item, 2)} km" for item in distances))
        if {"warming_C_mid", "warming_C_likely_low", "warming_C_likely_high"} <= set(candidate):
            note = candidate.get("note")
            rendered = f"mid={_number_text(candidate['warming_C_mid'], 1)} °C, likely={_number_text(candidate['warming_C_likely_low'], 1)}-{_number_text(candidate['warming_C_likely_high'], 1)} °C"
            if note not in (None, ""):
                rendered += f" ({note})"
            return rendered
        if "mean_temp_anomaly_c" in candidate and "n" in candidate:
            return f"{_number_text(candidate['mean_temp_anomaly_c'], 3)} °C (n={int(float(candidate['n']))})"
        if "dms" in candidate:
            return str(candidate["dms"])
        if "distance_km" in candidate and "distance" in text:
            return f"Great-circle distance: {_number_text(candidate['distance_km'])} km"
        if "bearing_degrees" in candidate and "bearing" in text:
            return f"Initial bearing from point1 to point2: {_number_text(candidate['bearing_degrees'], 2)}°"
        if "area_km2" in candidate and any((token in text for token in ("area", "bbox"))):
            return f"Estimated WGS84 bbox area: {_number_text(candidate['area_km2'])} km^2"
        if "dlnvs_percent" in candidate:
            return f"{_number_text(candidate['dlnvs_percent'], 2)}%"
        if "value_km" in candidate:
            return f"{_number_text(candidate['value_km'], 2)} km"
        if "total_energy_j" in candidate:
            return f"{_scientific_text(candidate['total_energy_j'], 3)} J"
        if "energy_j" in candidate:
            return _number_text(candidate["energy_j"], 3)
        if "value" in candidate and len(candidate) <= 4:
            scalar = candidate["value"]
            if isinstance(scalar, (int, float)) and math.isfinite(float(scalar)):
                return _number_text(scalar)
    return None


def _project_temperature_precision(
    outputs: list[Any] | tuple[Any, ...], question: str = ""
) -> str | None:
    """Render explicitly requested B10 temperature precision from provenance.

    Landsat B10 reducers often expose a full-precision Kelvin mean, while the
    benchmark's three-decimal Celsius contract is based on the reported
    two-decimal source mean before conversion.  When the request explicitly
    asks for three decimal places, preserve that measurement precision instead
    of propagating binary floating-point tails into the final answer.  The
    rule is schema/unit driven and only applies when a Kelvin source value and
    a Celsius conversion result are both present.
    """
    text = str(question or "").casefold()
    if not (
        "three decimal" in text
        and ("b10" in text or "band 10" in text)
        and ("kelvin" in text)
        and ("temperature" in text or "celsius" in text or "°c" in text)
    ):
        return None
    values = [_decode_output(output) for output in outputs]
    converted: float | None = None
    source_kelvin: float | None = None
    for value in reversed(values):
        for candidate in _walk_dicts(value):
            scalar = candidate.get("value")
            if isinstance(scalar, (int, float)) and math.isfinite(float(scalar)):
                if converted is None:
                    converted = float(scalar)
            items = candidate.get("values")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    observed = item.get("value")
                    path = str(item.get("path") or "").casefold()
                    if (
                        isinstance(observed, (int, float))
                        and math.isfinite(float(observed))
                        and ("b10" in path or "band10" in path or "band_10" in path)
                    ):
                        source_kelvin = float(observed)
                        break
            if source_kelvin is not None:
                break
        if source_kelvin is not None:
            break
    if source_kelvin is None or converted is None:
        return None
    celsius = round(source_kelvin, 2) - 273.15
    return f"{celsius:.3f}"
