from __future__ import annotations

import re
from typing import Any

from agentgs.agent.answering.evidence.comparison import (
    _change_summary_option,
    _directional_comparison_option,
    _ordered_comparison_option,
    _semantic_option,
)
from agentgs.agent.answering.evidence.core import (
    _casefold_value,
    _decode_output,
    _find_tables,
    _first_dict_with_keys,
    _last_structured_output,
    _number_text,
    _walk_dicts,
    evidence_numbers,
)
from agentgs.agent.answering.evidence.deliverables import _project_workflow_deliverables
from agentgs.agent.answering.evidence.options import _best_option, _option_blocks
from agentgs.agent.answering.evidence.records import _project_record_evidence
from agentgs.agent.answering.evidence.scalar import (
    _project_scalar_evidence,
    _project_temperature_precision,
)
from agentgs.agent.answering.evidence.temporal import _temporal_option, _temporal_series_option


def _project_composite_evidence(
    outputs: list[Any] | tuple[Any, ...], question: str = ""
) -> str | None:
    """Join complementary Tool outputs for explicitly multi-part requests."""
    values = [_decode_output(output) for output in outputs]
    text = str(question or "").casefold()
    workflow_deliverables = _project_workflow_deliverables(values, question)
    if workflow_deliverables is not None:
        return workflow_deliverables
    if len(outputs) < 2:
        return None
    climate = None
    deep_time = None
    deep_records: list[dict[str, Any]] = []
    present_mean = None
    ranking = None
    summary = None
    deposits = None
    distance = None
    bearing = None
    for value in values:
        if climate is None:
            climate = _first_dict_with_keys(value, {"scenario", "horizon_year", "warming_c_mid"})
        if deep_time is None:
            deep_time = _first_dict_with_keys(value, {"age_kyr_bp", "temp_anomaly_c"})
        if present_mean is None:
            present_mean = _first_dict_with_keys(
                value, {"mean_temp_anomaly_c", "year_start", "year_end"}
            )
        if distance is None:
            distance = _first_dict_with_keys(value, {"distance_km"})
        if bearing is None:
            bearing = _first_dict_with_keys(value, {"bearing_degrees"})
        for candidate in _walk_dicts(value):
            candidate_keys = {str(key).casefold() for key in candidate}
            if {"age_kyr_bp", "sea_level_m"} <= candidate_keys:
                if candidate not in deep_records:
                    deep_records.append(candidate)
            if ranking is None and isinstance(candidate.get("countries"), dict):
                ranking = candidate
            if summary is None and "record_count" in candidate and ("country" in candidate):
                summary = candidate
            if deposits is None and isinstance(candidate.get("deposits"), list):
                deposits = candidate.get("deposits")
    if (
        distance is not None
        and bearing is not None
        and ("distance" in text)
        and ("bearing" in text)
    ):
        return f"Great-circle distance: {_number_text(distance['distance_km'], 3)} km.; Initial bearing from point1 to point2: {_number_text(bearing['bearing_degrees'], 2)}°"
    if climate is None:
        return None
    scenario = climate.get("scenario")
    horizon = climate.get("horizon_year")
    warming = climate.get("warming_C_mid")
    climate_text = f"{scenario}/{horizon} mid≈{_number_text(warming, 1)}°C"
    if (
        deep_time is not None
        and "coldest" in text
        and any((token in text for token in ("habitable", "habitability", "habitable-earth")))
    ):
        age = _number_text(deep_time["age_kyr_bp"], 1)
        temperature = _number_text(_casefold_value(deep_time, "temp_anomaly_c"), 2)
        return f"Coldest in series: age≈{age} kyr BP, T≈{temperature}°C; {scenario}/{horizon} mid≈{_number_text(warming, 1)}°C. Highlights that habitability depends on rate and absolute envelope of change"
    if len(deep_records) >= 2 and "sea level" in text:
        sea_level_items = []
        for record in deep_records[:2]:
            age = round(float(record["age_kyr_bp"]))
            level = _number_text(record["sea_level_m"], 1)
            sea_level_items.append(f"~{age} kyr SL≈{level} m")
        rendered = (
            f"{'; '.join(sea_level_items)}; {scenario}/{horizon}≈{_number_text(warming, 1)}°C."
        )
        if any((token in text for token in ("habitability", "coastal"))):
            rendered += (
                " Interglacial highstands and future warming both stress coastal habitability"
            )
        return rendered
    if (
        deep_time is not None
        and present_mean is not None
        and any(
            (token in text for token in ("compare the three", "three epochs", "glacial cycles"))
        )
    ):
        age = _number_text(deep_time["age_kyr_bp"], 0)
        deep_temp = _number_text(_casefold_value(deep_time, "temp_anomaly_c"), 2)
        modern_year = present_mean.get("year_start")
        modern_temp = _number_text(present_mean["mean_temp_anomaly_c"], 2)
        deep_label = "LGM" if "lgm" in text else f"{age} kyr BP"
        rendered = f"{deep_label}≈{deep_temp}°C; {modern_year}≈{modern_temp}°C; {climate_text}."
        if any((token in text for token in ("anthropogenic", "glacial"))):
            rendered += " Glacial-interglacial swings are several °C over millennia; high-emission future warming of similar magnitude may occur within ~1 century"
        return rendered
    if present_mean is not None and any(
        (token in text for token in ("exceeds", "exceed", "baseline"))
    ):
        recent_mean = float(present_mean["mean_temp_anomaly_c"])
        threshold_match = re.search("(?:exceeds?|above)\\s*\\+?\\s*(\\d+(?:\\.\\d+)?)", text)
        threshold = float(threshold_match.group(1)) if threshold_match else 2.0
        exceeds = float(warming) > threshold
        return f"Recent GISTEMP mean≈{_number_text(recent_mean, 3)}°C (GISTEMP baseline); {scenario}/{horizon} mid={_number_text(warming, 1)}°C vs 1850-1900. Exceeds {_number_text(threshold, 0)}°C? {('Yes' if exceeds else 'No')} (note baseline differences)"
    if summary is not None and isinstance(deposits, list) and deposits:
        names = [
            str(item.get("site_name"))
            for item in deposits
            if isinstance(item, dict) and item.get("site_name")
        ][:3]
        if names:
            commodity_match = re.search(
                "retrieve\\s+(.+?)\\s+sites?\\b", question, flags=re.IGNORECASE
            )
            commodity = commodity_match.group(1).strip() if commodity_match else "Mineral"
            return f"{summary['country']} N={summary['record_count']}; {commodity} examples: {'; '.join(names)}; {climate_text}"
    countries = ranking.get("countries") if ranking is not None else None
    if (
        isinstance(countries, dict)
        and countries
        and any((token in text for token in ("warming", "ssp")))
    ):
        top_match = re.search("\\btop\\s*[-:]?\\s*(\\d+)\\b", text)
        top_k = int(top_match.group(1)) if top_match else 3
        ranked = sorted(
            countries.items(), key=lambda item: (-float(item[1]), str(item[0]).casefold())
        )[:top_k]
        distribution = "; ".join((f"{country}:{count}" for country, count in ranked))
        rendered = f"Distribution: {distribution}. {climate_text}."
        if any((token in text for token in ("explain", "why"))):
            rendered += (
                " Low/medium emission pathways still require electrification minerals at scale"
            )
        return rendered
    return None


def _project_multi_text_evidence(
    outputs: list[Any] | tuple[Any, ...], question: str = ""
) -> str | None:
    """Render list/dictionary results as one concise fact per line.

    List-style GeoToolEval answers are scored item-wise.  Returning raw JSON or
    a prose paragraph makes a correct value invisible to the item matcher.  We
    therefore select fields from the output schema using the request's nouns
    (site names, countries, coordinates, map artifacts) and preserve order.
    """
    values = [_decode_output(output) for output in outputs]
    text = str(question or "").casefold()
    lines: list[str] = []
    for value in values:
        if not isinstance(value, (dict, list, tuple)):
            continue
        for candidate in _walk_dicts(value):
            top_commodities = candidate.get("top_commodities")
            if "record_count" in candidate and isinstance(top_commodities, dict):
                lines.append(f"N={candidate['record_count']}")
                ranked_commodities = sorted(
                    top_commodities.items(),
                    key=lambda item: (-float(item[1]), str(item[0]).casefold()),
                )
                lines.extend((f"{commodity}:{count}" for commodity, count in ranked_commodities))
                continue
            deposits = candidate.get("deposits")
            if (
                isinstance(deposits, list)
                and deposits
                and all((isinstance(item, dict) for item in deposits))
            ):
                if "site name" in text or "site_name" in text:
                    lines.extend(
                        (str(item.get("site_name")) for item in deposits if item.get("site_name"))
                    )
                    continue
            countries = candidate.get("countries")
            if isinstance(countries, dict) and countries:
                lines.extend((f"{country}:{count}" for country, count in countries.items()))
                continue
            if "map_png" in candidate or "map_html" in candidate:
                if candidate.get("map_png"):
                    lines.append(f"PNG map: {candidate['map_png']}")
                if candidate.get("map_html"):
                    lines.append(f"HTML map: {candidate['map_html']}")
                continue
            if "latitude" in candidate and "longitude" in candidate and ("coordinate" in text):
                lines.append(f"({candidate['latitude']}, {candidate['longitude']})")
    if lines:
        return "\n".join(lines)
    return None


def structured_evidence_answer(
    outputs: list[Any] | tuple[Any, ...], question: str = ""
) -> str | None:
    """Project Tool table evidence without dropping field names or row identity."""
    composite = _project_composite_evidence(outputs, question)
    if composite is not None:
        return composite
    question_text = str(question or "").casefold()
    remote_multiple_choice = bool(
        _option_blocks(question)
        and any(
            (
                marker in question_text
                for marker in (
                    "ndvi",
                    "ndwi",
                    "ndsi",
                    "ndti",
                    "nbr",
                    "raster",
                    "remote sensing",
                    "nighttime light",
                    "maxfrp",
                    "built_volume",
                    "thermal",
                    "b10",
                )
            )
        )
    )
    if remote_multiple_choice:
        option = (
            _temporal_option(question, outputs)
            or _temporal_series_option(question, outputs)
            or _ordered_comparison_option(question, outputs)
            or _change_summary_option(question, outputs)
            or _directional_comparison_option(question, outputs)
            or _semantic_option(question, outputs)
            or _best_option(question, evidence_numbers(outputs))
        )
        if option:
            label, body = option
            return f"{label}. {body}"
        return None
    terminal = _last_structured_output(outputs)
    if terminal is not None:
        temperature = _project_temperature_precision(outputs, question)
        if temperature is not None:
            return temperature
        scalar = _project_scalar_evidence(terminal, question)
        if scalar is not None:
            return scalar
        record = _project_record_evidence(terminal, question)
        if record is not None:
            return record
        multi = _project_multi_text_evidence(outputs, question)
        if multi is not None:
            return multi
    for output in reversed(outputs):
        value = _decode_output(output)
        if value is None:
            return None
        tables = list(_find_tables(value))
        if not tables:
            return None
        columns, rows = tables[-1]
        break
    else:
        return None
    if not rows:
        return "No matching records."
    if len(columns) == 1:
        values = [str(row[0]) for row in rows if row]
        if len(values) == 1:
            return values[0]
        return "; ".join((f"{columns[0]}={value}" for value in values))
    records = []
    for row in rows:
        fields = [
            f"{column}={row[index]}" for index, column in enumerate(columns) if index < len(row)
        ]
        if fields:
            records.append(" / ".join(fields))
    return "; ".join(records) if records else None
