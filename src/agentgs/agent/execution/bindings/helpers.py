"""Parameter parsing, cohort grouping and scientific reference helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from agentgs.contracts.resources import RESOURCE_MARKER as RESOURCE_MARKER
from agentgs.contracts.resources import split_question_resource as split_question_resource

_INVALID_VALUES = (None, "", "*", "null", "None")
_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_generated(value: str) -> dict[str, Any]:
    text = (value or "").strip()
    text = re.sub("^(?:input\\s*:)", "", text, flags=re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(
            "^```(?:json)?\\s*|\\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL
        ).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"input_text": text} if text else {}
    if isinstance(parsed, dict):
        return _flatten_wrappers(parsed)
    if isinstance(parsed, list):
        return {"x": parsed}
    return {"value": parsed}


def _flatten_wrappers(value: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(value)
    for key in ("source_locator", "arguments", "parameters", "params"):
        nested = value.get(key)
        if isinstance(nested, dict):
            for nested_key, nested_value in nested.items():
                if flattened.get(nested_key) in _INVALID_VALUES:
                    flattened[nested_key] = nested_value
    nested_input = value.get("input")
    if isinstance(nested_input, dict):
        for nested_key, nested_value in nested_input.items():
            flattened.setdefault(nested_key, nested_value)
    return flattened


def _resource_object(resource: Any) -> dict[str, Any]:
    if not isinstance(resource, dict):
        return {}
    flattened = _flatten_wrappers(resource)
    flattened.pop("source_locator", None)
    return flattened


def _named_query_parameters(question: str) -> dict[str, Any]:
    """Extract explicit ``name=value`` literals without consulting benchmark labels."""
    extracted: dict[str, Any] = {}
    pattern = re.compile(
        "\\b([A-Za-z][A-Za-z0-9_]*)\\s*=\\s*(\\[[^\\]\\r\\n]*\\]|\\{[^\\}\\r\\n]*\\}|\\\"[^\\\"\\r\\n]*\\\"|'[^'\\r\\n]*'|[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?|[A-Za-z0-9_.:/-]+)"
    )
    for match in pattern.finditer(question or ""):
        key, raw_value = match.groups()
        value: Any = raw_value.strip()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            if len(value) >= 2 and value[0] == value[-1] and (value[0] in {"'", '"'}):
                value = value[1:-1]
        extracted[key] = value
    return extracted


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = start + consumed
    return objects


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key == "_rs_contract"
                and isinstance(item, dict)
                and (item.get("version") == "rs-local-v1")
            ):
                continue
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def _paths_from(values: Iterable[Any]) -> list[str]:
    paths: list[str] = []
    for value in values:
        for item in _walk(value):
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if re.search("\\.(?:csv|geojson|json|jsonl|tif|tiff|npy)$", candidate, re.IGNORECASE):
                paths.append(candidate)
    return list(dict.fromkeys(paths))


def _raster_files(objects: list[dict[str, Any]]) -> list[str]:
    paths = _paths_from(objects)
    return [path for path in paths if Path(path).suffix.casefold() in (".tif", ".tiff")]


def _scene_key(path: str) -> str:
    stem = Path(path).stem.casefold()
    stem = re.sub(
        "(?:qa[_ -]?pixel|sr[_ -]?b\\d+|band[_ -]?\\d+|b\\d+|green|nir|red|swir\\d*)", "", stem
    )
    return re.sub("[^a-z0-9]", "", stem)


def _pair_scene_files(first_paths: list[str], second_paths: list[str]) -> list[tuple[str, str]]:
    second_by_scene: dict[str, list[str]] = {}
    for path in second_paths:
        second_by_scene.setdefault(_scene_key(path), []).append(path)
    pairs: list[tuple[str, str]] = []
    for first in first_paths:
        matches = second_by_scene.get(_scene_key(first)) or []
        if matches:
            pairs.append((first, matches[0]))
    if not pairs and len(first_paths) == len(second_paths):
        pairs = list(zip(sorted(first_paths), sorted(second_paths)))
    return pairs


def _band_numbers(text: str) -> set[int]:
    return {
        int(value)
        for value in re.findall(
            "(?<![a-z0-9])(?:(?:sur[_ -]?refl|sr)[_ -]?)?b(?:and)?[_ -]?(\\d{1,2})(?!\\d)",
            text,
            re.IGNORECASE,
        )
    }


def _path_has_band(path: str, band: int) -> bool:
    return bool(
        re.search(
            f"(?:^|[_ -])(?:sr[_ -]?)?b(?:and)?[_ -]?0?{band}(?:[_ .-]|$)",
            Path(path).name,
            re.IGNORECASE,
        )
    )


def _role_band(clean_question: str, role_tokens: tuple[str, ...]) -> int | None:
    band_matches = list(
        re.finditer(
            "(?<![a-z0-9])(?:(?:sur[_ -]?refl|sr)[_ -]?)?b(?:and)?[_ -]?(\\d{1,2})(?!\\d)",
            clean_question,
            re.IGNORECASE,
        )
    )
    role_positions = [
        match.start()
        for token in role_tokens
        for match in re.finditer(
            f"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", clean_question, re.IGNORECASE
        )
    ]
    if not band_matches or not role_positions:
        return None
    nearest = min(
        band_matches,
        key=lambda match: min((abs(match.start() - position) for position in role_positions)),
    )
    return int(nearest.group(1))


def _numeric_series(objects: list[dict[str, Any]]) -> list[list[float]]:
    series: list[list[float]] = []
    for obj in objects:
        values = obj.get("values")
        if isinstance(values, list):
            extracted = []
            for item in values:
                candidate = item.get("value") if isinstance(item, dict) else item
                if isinstance(candidate, (int, float)):
                    extracted.append(float(candidate))
            if extracted:
                series.append(extracted)
        for key in ("x", "y"):
            value = obj.get(key)
            if (
                isinstance(value, list)
                and value
                and all((isinstance(item, (int, float)) for item in value))
            ):
                series.append([float(item) for item in value])
        scalar = obj.get("value")
        if isinstance(scalar, (int, float)):
            series.append([float(scalar)])
        semantic_values: list[float] = []
        for item in _walk(obj):
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if str(key).casefold() in {
                    "temp_anomaly_c",
                    "warming_c_mid",
                    "mid",
                    "mean",
                    "mean_temp_anomaly_c",
                    "dlnvs_percent",
                    "value_km",
                    "capacity_mw",
                    "distance_km",
                } and isinstance(value, (int, float)):
                    semantic_values.append(float(value))
        if semantic_values:
            series.append(semantic_values)
    return series


def _named_value_series(objects: list[dict[str, Any]], label: str) -> list[float]:
    matched: list[tuple[str, float]] = []
    normalized_label = label.casefold()
    for obj in objects:
        values = obj.get("values")
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            value = item.get("value")
            if normalized_label in Path(path).name.casefold() and isinstance(value, (int, float)):
                matched.append((path, float(value)))
    return [value for _, value in sorted(matched)]


def _set_if_missing(params: dict[str, Any], key: str, value: Any) -> None:
    if value not in _INVALID_VALUES and params.get(key) in _INVALID_VALUES:
        params[key] = value


def _partition_sequence(
    values: list[Any], occurrence_index: int, occurrence_count: int
) -> list[Any]:
    """Select a deterministic cohort for repeated Tool instances.

    The partition is based on scene/entity boundaries when possible and falls
    back to balanced contiguous chunks.  It is generic for arbitrary resource
    lists and does not inspect evaluation identifiers.
    """
    if occurrence_count <= 1 or len(values) <= 1:
        return list(values)
    occurrence_index = max(0, min(int(occurrence_index), occurrence_count - 1))
    grouped: dict[str, list[Any]] = {}
    years = {
        match.group(1)
        for value in values
        for match in [re.search("(?:^|[_ -])(\\d{4})(?:[-_]|$)", Path(str(value)).stem)]
        if match
    }
    for value in values:
        text = str(value)
        stem = Path(text).stem.casefold()
        year_match = re.search("(?:^|[_ -])(\\d{4})(?:[-_]|$)", stem)
        stem = re.sub(
            "(?:[_ -](?:sr[_ -]?)?b(?:and)?\\d+|[_ -](?:qa[_ -]?)?pixel|[_ -]\\d{4}[-_]?\\d{2}[-_]?\\d{2}|[_ -]\\d{4})",
            "",
            stem,
        )
        stem = re.sub("[^a-z0-9\\u3400-\\u9fff]+", "_", stem).strip("_")
        if len(years) > 1 and year_match:
            stem = f"{stem}__year{year_match.group(1)}"
        grouped.setdefault(stem or "default", []).append(value)
    groups = [items for _, items in sorted(grouped.items()) if items]
    if len(groups) >= occurrence_count:
        return list(groups[occurrence_index % len(groups)])
    size = max(1, (len(values) + occurrence_count - 1) // occurrence_count)
    start = occurrence_index * size
    return list(values[start : min(len(values), start + size)]) or list(values[-size:])


def _semantic_period_partition(
    values: list[str], question: str, occurrence_index: int, occurrence_count: int
) -> list[str]:
    """Partition a raster series by periods explicitly named in the request."""
    lowered_question = str(question or "").casefold()
    source_families = _source_raster_groups(values)
    largest_family = max((len(items) for items in source_families), default=0)
    coherent_families = [
        items for items in source_families if len(items) == largest_family and items
    ]
    if len(coherent_families) >= occurrence_count and occurrence_count > 1:
        return list(coherent_families[min(occurrence_index, len(coherent_families) - 1)])
    requested_months = list(
        dict.fromkeys(
            (
                month
                for _, month in sorted(
                    (
                        (match.start(), month)
                        for name, month in _MONTH_NUMBERS.items()
                        for match in re.finditer(f"\\b{name}\\b", lowered_question)
                    )
                )
            )
        )
    )
    month_comparison = any(
        (
            phrase in lowered_question
            for phrase in (
                "which month",
                "compare the month",
                "compare monthly",
                "difference between the two",
                "each month",
            )
        )
    )
    if len(requested_months) >= occurrence_count and occurrence_count > 1 and month_comparison:
        by_month: dict[int, list[str]] = {}
        requested = set(requested_months)
        for value in values:
            match = re.search(
                "(?:^|[_ -])(?:19|20)\\d{2}[-_](\\d{2})(?:[-_]|$)", Path(str(value)).stem
            )
            if match and int(match.group(1)) in requested:
                by_month.setdefault(int(match.group(1)), []).append(value)
        ordered_months = [by_month[month] for month in requested_months if by_month.get(month)]
        if len(ordered_months) >= occurrence_count:
            return list(ordered_months[min(occurrence_index, len(ordered_months) - 1)])
    requested_quarters = list(
        dict.fromkeys(re.findall("\\bq([1-4])\\b", question or "", re.IGNORECASE))
    )
    if len(requested_quarters) >= occurrence_count:
        by_quarter: dict[str, list[str]] = {}
        requested = set(requested_quarters)
        for value in values:
            match = re.search(
                "(?:^|[_ -])(?:19|20)\\d{2}[-_](\\d{2})(?:[-_]|$)", Path(str(value)).stem
            )
            if not match:
                continue
            quarter = str((int(match.group(1)) - 1) // 3 + 1)
            if quarter in requested:
                by_quarter.setdefault(quarter, []).append(value)
        ordered = [by_quarter[q] for q in requested_quarters if by_quarter.get(q)]
        if len(ordered) >= occurrence_count:
            return list(ordered[min(occurrence_index, len(ordered) - 1)])
    return _partition_sequence(values, occurrence_index, occurrence_count)


def _source_raster_groups(values: list[str]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for value in values:
        stem = Path(str(value)).stem.casefold()
        signature = re.sub("(?:19|20)\\d{2}(?:[-_]\\d{1,2}){0,2}", "", stem)
        signature = re.sub("[^a-z0-9]+", "_", signature).strip("_") or "default"
        groups.setdefault(signature, []).append(value)
    return [items for _, items in sorted(groups.items()) if items]


def _year_partition(values: list[str], occurrence_index: int, occurrence_count: int) -> list[str]:
    by_year: dict[str, list[str]] = {}
    for value in values:
        match = re.search("(?:^|[_ -])(\\d{4})(?:[-_]|$)", Path(str(value)).stem)
        by_year.setdefault(match.group(1) if match else "unknown", []).append(value)
    groups = [items for _, items in sorted(by_year.items()) if items]
    if len(groups) >= occurrence_count:
        return list(groups[min(occurrence_index, len(groups) - 1)])
    return list(values)


def _temporal_source_rasters(
    values: list[str], occurrence_index: int, occurrence_count: int
) -> list[str]:
    """Choose temporal input stacks without confusing a second variable.

    Resource directories can contain several raster families (for example a
    light-intensity time series and a building-volume layer).  Repeated
    temporal reducers should consume the largest coherent family, then split
    it by year/entity.  The choice is based only on filename structure and
    provenance, never on benchmark answers.
    """
    if not values:
        return []
    groups = _source_raster_groups(values)
    if not groups:
        return list(values)
    largest = max((len(items) for items in groups))
    candidates = [items for items in groups if len(items) == largest]
    if len(candidates) >= occurrence_count:
        return list(candidates[min(occurrence_index, len(candidates) - 1)])
    selected = candidates[0] if candidates else values
    by_year = _year_partition(selected, occurrence_index, occurrence_count)
    if by_year != list(selected):
        return by_year
    return _partition_sequence(selected, occurrence_index, occurrence_count)


def _chronological_primary_rasters(values: list[str]) -> list[str]:
    """Return the largest coherent raster family in chronological order."""
    groups = _source_raster_groups(values)
    if not groups:
        return sorted(values)
    selected = max(groups, key=len)

    def sort_key(value: str) -> tuple[str, str]:
        match = re.search("((?:19|20)\\d{2})[-_](\\d{2})[-_](\\d{2})", Path(str(value)).stem)
        return ("-".join(match.groups()) if match else "", str(value))

    return sorted(selected, key=sort_key)


def _select_numeric_series(
    series: list[list[float]], occurrence_index: int, occurrence_count: int
) -> list[float] | None:
    if not series:
        return None
    if occurrence_count <= 1:
        return series[-1]
    candidates = series[-occurrence_count:]
    return candidates[min(occurrence_index, len(candidates) - 1)]


def _artifact_producer_series(artifact_registry: Any, *, producer: str) -> list[list[float]]:
    """Return one primary numeric series per structured producer result.

    The rendered transcript contains historical Tool inputs as well as outputs.
    A repeated reducer can therefore mistake the first reducer's input for a new
    temporal cohort. Structured artifact records contain output provenance only.
    """
    if artifact_registry is None:
        return []
    results: list[list[float]] = []
    try:
        records = artifact_registry.records
    except Exception:
        return []
    for record in records:
        if (
            getattr(record, "producer", None) != producer
            or getattr(record, "semantic_type", None) != "StructuredResult"
        ):
            continue
        value = getattr(record, "value", None)
        if not isinstance(value, dict):
            continue
        candidates = _numeric_series([value])
        if candidates:
            results.append(max(candidates, key=len))
    return results


def tool_input_is_ready(tool_name: str, normalized_input: str) -> bool:
    """Return whether deterministic artifacts satisfy a known Tool contract."""
    try:
        params = json.loads(normalized_input)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(params, dict):
        return False
    if tool_name.startswith("RemoteSensing") and params.get("_rs_binding_issues"):
        return False
    required_any = {
        "RemoteSensingListFiles": (("dir_path",),),
        "RemoteSensingBatchImageMean": (("file_list",),),
        "RemoteSensingBatchImageSum": (("file_list",),),
        "RemoteSensingTiffAverage": (("file_list",),),
        "RemoteSensingBatchHotspotPercentage": (("file_list", "threshold"),),
        "RemoteSensingBatchHotspotRaster": (("file_list", "threshold"),),
        "RemoteSensingBatchNDSI": (("green_file_list", "swir_file_list"),),
        "RemoteSensingNDWI": (
            ("input_swir_paths", "input_nir_paths"),
            ("input_swir_path", "input_nir_path"),
        ),
        "RemoteSensingNBR": (
            ("input_nir_paths", "input_swir_paths"),
            ("input_nir_path", "input_swir_path"),
        ),
        "RemoteSensingCloudMask": (
            ("sr_band_paths", "qa_pixel_paths"),
            ("sr_band_path", "qa_pixel_path"),
        ),
        "RemoteSensingNDTI": (("input_pairs",),),
        "RemoteSensingMean": (("x",),),
        "RemoteSensingMaxValueAndIndex": (("x",),),
        "RemoteSensingMinValueAndIndex": (("x",),),
        "RemoteSensingCoefficientOfVariation": (("x",),),
        "RemoteSensingKurtosis": (("x",),),
        "RemoteSensingSensSlope": (("x",),),
        "RemoteSensingMannKendallTest": (("x",),),
        "RemoteSensingArgmax": (("x",),),
        "RemoteSensingLinearTrend": (("y",),),
        "RemoteSensingDifference": (("a", "b"),),
        "RemoteSensingPercentageChange": (("a", "b"),),
        "RemoteSensingDivision": (("a", "b"), ("a_values", "b_values")),
        "DeepEarthQueryLITHO1": (("location_key", "layer"),),
        "DeepEarthQuerySEMUCB": (("location_key", "depth_km"),),
        "DeepEarthQueryUSGSCatalog": (("operation",),),
        "DeepEarthGetUSGSEvent": (("event_id",),),
        "DeepEarthSampleEventModels": (("event_id",),),
        "GeoBigDataExecuteGeoSPARQL": (("query",),),
        "MineralResourcesSearchDeposits": (("country",), ("commodity",)),
        "MineralResourcesGetDeposit": (("dep_id",),),
        "MineralResourcesCountrySummary": (("country",),),
        "MineralResourcesRankCommodityCountries": (("commodity",),),
        "ClimateQueryDeepTime": (("dataset", "age_kyr_bp"), ("dataset", "ages_kyr_bp"), ("find",)),
        "ClimateQueryPresentMean": (("year_start",), ("year_start", "year_end")),
        "ClimateQueryFutureSSP": (("scenario", "horizon_year"), ("ssp", "ssp_horizon")),
        "ClimateCompareEpochs": (("values",), ("x",)),
        "EnergyFacilityDistance": (("country",),),
        "EnergyMineralSiteDistance": (("dep_id_1", "dep_id_2"),),
        "EnergyBBoxArea": (("bbox",),),
        "EnergyNearestFacility": (("country", "energy_source"),),
        "EnergyCapacityValue": (("series", "year"),),
        "EnergyNearestMineralSite": (("plant_name",),),
        "EnergyCountryRenewableSummary": (("country",),),
        "EnergyCapacityGrowth": (("series", "start_year", "end_year"),),
        "EnergyTransitionMetrics": (("country",),),
        "EnergyRegionalOverlaySummary": (("region", "bbox"),),
        "OceanMonthlyGridValue": (("date", "latitude", "longitude"),),
        "OceanProfileValue": (("expocode", "station"),),
        "OceanCruiseQC": (("cruise",),),
        "OceanStationDistance": (("glodap_expocode", "glodap_station", "snapo_cruise"),),
        "OceanBBoxArea": (("bbox",),),
        "OceanAnnualCarbonSummary": (("year",),),
        "OceanVerticalProfileDifference": (("expocode", "station"),),
        "OceanRegionalMapSummary": (("region", "bbox"),),
        "OceanSurfaceDeepMatch": (("glodap_expocode", "glodap_station"),),
        "OceanProfileDeduplication": (("cruise",),),
        "DeepTimeDecimalToDMS": (("value", "field"), ("input_text",)),
        "DeepTimeHaversineDistance": (
            ("lat1", "lon1", "lat2", "lon2"),
            ("series_ids",),
            ("input_text",),
        ),
        "DeepTimeBearing": (("lat1", "lon1", "lat2", "lon2"), ("series_ids",), ("input_text",)),
        "DeepTimeBBoxArea": (("bbox",), ("input_text",)),
        "CCSDecimalToDMS": (("value", "field"), ("case_id",), ("facility_id",), ("input_text",)),
        "CCSHaversineDistance": (("lat1", "lon1", "lat2", "lon2"), ("case_ids",), ("input_text",)),
        "CCSBearing": (("lat1", "lon1", "lat2", "lon2"), ("case_ids",), ("input_text",)),
        "CCSBBoxArea": (("bbox",), ("input_text",)),
        "CCSCheckCRSConsistency": (("crs_values",), ("input_assets",), ("input_text",)),
        "RemoteSensingNDVI": (
            ("nir_band_files", "red_band_files"),
            ("nir_band_file", "red_band_file"),
        ),
        "RemoteSensingCountAboveThreshold": (("file_path",),),
        "RemoteSensingTiffDifference": (
            ("image_a_paths", "image_b_paths"),
            ("image_a_path", "image_b_path"),
        ),
        "RemoteSensingExtremeSnowLossPercentage": (("binary_map_path",),),
        "RemoteSensingSubtract": (("img1_path", "img2_path"),),
        "RemoteSensingFireIncreaseMap": (("change_image_path", "threshold"),),
        "RemoteSensingKelvinToCelsius": (("kelvin",),),
        "NaturalHazardLoadEventTable": (("source", "sheet"),),
        "NaturalHazardFilterRecords": (("source", "sheet", "filters"),),
        "NaturalHazardCountRecords": (("filtered_records_path",), ("source", "sheet", "filters")),
        "NaturalHazardSumField": (("measure", "filters"),),
        "NaturalHazardGroupAggregate": (("group_by", "filters"),),
        "NaturalHazardSelectMaxRecord": (("measure", "filters"),),
        "NaturalHazardBuildStartDate": (("filters",),),
        "NaturalHazardSelectLatest": (("filters",),),
    }
    for prefix in ("DeepTime", "CCS"):
        required_any.update(
            {
                f"{prefix}StandardizeSpatialData": (("paths",), ("input_assets",)),
                f"{prefix}ValidateSpatialData": (("paths",), ("input_assets",)),
                f"{prefix}NearestNeighbor": (("paths",),),
                f"{prefix}Buffer": (("paths",),),
                f"{prefix}Overlay": (("paths",),),
                f"{prefix}BatchVector": (("paths",), ("input_assets",)),
                f"{prefix}InterpolateIDW": (("paths",),),
                f"{prefix}RasterAlgebra": (("paths",),),
                f"{prefix}ZonalStatistics": (("paths",),),
                f"{prefix}RenderMap": (("paths",), ("input_assets",), ("title",)),
            }
        )
    alternatives = required_any.get(tool_name)
    if not alternatives:
        return False
    for fields in alternatives:
        if all(
            (
                params.get(field) not in _INVALID_VALUES and params.get(field) != []
                for field in fields
            )
        ):
            return True
    return False
