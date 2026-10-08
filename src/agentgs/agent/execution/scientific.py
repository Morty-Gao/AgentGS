"""Expose preserved scientific contracts to the historical Agent argument planner.

Only eight named spatial tools are affected. Catalog loading is data-only; this
module does not inspect benchmark questions/answers, read observations, choose a
measurement, fabricate a region, or execute a scientific backend.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from functools import lru_cache
from typing import Any, Mapping

_SUFFIXES = ("InterpolateIDW", "RasterAlgebra", "ZonalStatistics", "RenderMap")
SCIENTIFIC_TOOL_NAMES = frozenset(
    (prefix + suffix for prefix in ("DeepTime", "CCS") for suffix in _SUFFIXES)
)
_CATALOG_PATH = None
_REQUIRED = {
    "InterpolateIDW": [
        "value_field",
        "points_path or point_path or exactly one paths entry",
        "date_field and month when a month is requested",
    ],
    "RasterAlgebra": [
        "raster_path or exactly one paths entry",
        "explicit operation",
        "scalar for add/subtract/multiply/divide",
    ],
    "ZonalStatistics": [
        "raster_path plus zones_path, OR table_path plus zone_field and value_field"
    ],
    "RenderMap": ["raster_path OR vector_path OR explicit spatial paths"],
}
_GUIDANCE = {
    "InterpolateIDW": "Choose the requested measured field only from actual supplied column/feature metadata. Never substitute the first numeric column, an identifier, or a guessed variable. Identify exactly one measured point layer. Select an explicitly requested month with date_field and month=YYYY-MM; do not mix months or invent a filter. Preserve the requested domain: this backend requires at least two distinct points and a valid increasing WGS84 bbox spanning at most 180 longitude degrees; do not silently crop or relabel a wider/global task.",
    "RasterAlgebra": "State the requested operation explicitly: mask, add, subtract, multiply or divide. Supply scalar for arithmetic or the intended minimum/maximum for a mask. Do not invent a mask or discard negative measurements merely because an old workflow suggested an algebra step. Identify exactly one upstream measurement raster and retain its georeference.",
    "ZonalStatistics": "Provide one georeferenced GeoTIFF and the actual polygon zone layer with unique zone identifiers, or one CSV with explicit zone_field/value_field. Missing zones cannot be replaced by global statistics. Raw NPY has no spatial transform. When several raster intermediates exist, explicitly choose the one representing the requested variable; do not select the first or latest by guess.",
    "RenderMap": "Provide actual georeferenced raster data or declared spatial features. A title is not input data. Raster maps use numeric values and must omit classification_field; classification_field is only for explicit vector attributes. Do not invent a field or render unrelated source layers when a requested result raster is absent.",
}
_OUTPUTS = {
    "InterpolateIDW": "raster_output (georeferenced GeoTIFF), value_field, CRS/bbox, finite_pixels and observation_selection when filtering was requested",
    "RasterAlgebra": "raster_output, operation and finite_pixels, retaining GeoTIFF spatial metadata",
    "ZonalStatistics": "zone_statistics (one record per real zone), zone_count and source provenance",
    "RenderMap": "map_png, map_html, actual spatial sources and finite_pixels or feature_count",
}


def _suffix(name: str) -> str:
    return next((suffix for suffix in _SUFFIXES if name.endswith(suffix)))


@lru_cache(maxsize=1)
def _catalog():
    from types import SimpleNamespace

    from agentgs.contracts.catalog import get_tool_spec

    return SimpleNamespace(get_tool_spec=get_tool_spec)


def requires_scientific_argument_generation(tool_name: str) -> bool:
    """Legacy {} -> resource paths/title is not a scientific parameter proof."""
    return tool_name in SCIENTIFIC_TOOL_NAMES


def overlay_scientific_tool_schema(
    tool_name: str, tool_info: Mapping[str, Any] | None
) -> dict[str, Any]:
    info = deepcopy(dict(tool_info or {}))
    if tool_name not in SCIENTIFIC_TOOL_NAMES:
        return info
    specification = _catalog().get_tool_spec(tool_name)
    if not specification or not specification.get("Input"):
        raise RuntimeError("Missing live scientific contract for " + tool_name)
    kind = _suffix(tool_name)
    info["input_format"] = (
        specification["Input"]
        + "\nRequired bindings: "
        + "; ".join(_REQUIRED[kind])
        + ".\nScientific argument constraints: "
        + _GUIDANCE[kind]
        + "\nUse only supplied observations, metadata and actual upstream outputs. If a required field, time selection or zone is unavailable, explain that missing input; do not fabricate a successful argument set."
    )
    info["output_format"] = (
        specification.get("Output", "JSON result")
        + ": "
        + _OUTPUTS[kind]
        + "; invalid scientific input returns status=error."
    )
    info["scientific_contract_source"] = (
        "live GeoEval catalog plus preserved scientific operation constraints"
    )
    info["scientific_required_bindings"] = list(_REQUIRED[kind])
    info["scientific_requires_generated_arguments"] = True
    return info


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        parsed = deepcopy(dict(value))
    else:
        if not isinstance(value, str):
            return {}
        text = re.sub("^input\\s*:", "", value.strip(), flags=re.I).strip()
        if text.startswith("```"):
            text = re.sub("^```(?:json)?\\s*|\\s*```$", "", text, flags=re.I)
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
    flattened = dict(parsed)
    for key in ("source_locator", "arguments", "parameters", "params"):
        nested = parsed.get(key)
        if isinstance(nested, dict):
            for field, field_value in nested.items():
                if flattened.get(field) in (None, "", "*", "null", "None"):
                    flattened[field] = deepcopy(field_value)
    if isinstance(parsed.get("input"), dict):
        for field, field_value in parsed["input"].items():
            flattened.setdefault(field, deepcopy(field_value))
    return flattened


def preserve_scientific_arguments(
    tool_name: str, generated_input: Any, normalized_input: Any
) -> Any:
    """Retain only model-declared arguments for the eight scientific operations.

    Run this after legacy prepare_tool_input and before any independently proven
    shared-bridge bindings. Even nested automatic locator metadata is excluded:
    the legacy backend recursively treats path-looking metadata as an input
    source. No path, measurement, date, zone or operation is synthesized here.
    """
    if tool_name not in SCIENTIFIC_TOOL_NAMES:
        return normalized_input
    return json.dumps(_object(generated_input), ensure_ascii=False)


def scientific_input_issues(tool_name: str, input_value: Any) -> list[str]:
    """Structural diagnostics only; actual field values/CRS remain backend checks."""
    if tool_name not in SCIENTIFIC_TOOL_NAMES:
        return []
    params = _object(input_value)
    issues: list[str] = []
    paths = params.get("paths")
    single_path = (
        isinstance(paths, list)
        and len(paths) == 1
        and isinstance(paths[0], str)
        and bool(paths[0].strip())
    )
    kind = _suffix(tool_name)
    if kind == "InterpolateIDW":
        if not isinstance(params.get("value_field") or params.get("field"), str) or not (
            params.get("value_field") or params.get("field")
        ):
            issues.append(
                "IDW needs an explicit measured value_field; identifiers must not be guessed."
            )
        if not (params.get("points_path") or params.get("point_path") or single_path):
            issues.append("IDW needs exactly one explicit point source, preferably points_path.")
        if params.get("month") is not None and (not params.get("date_field")):
            issues.append("Month selection needs an explicit date_field.")
    elif kind == "RasterAlgebra":
        if not (params.get("raster_path") or single_path):
            issues.append("Raster algebra needs exactly one explicit raster source.")
        operation = params.get("operation")
        if operation not in {"mask", "add", "subtract", "multiply", "divide"}:
            issues.append("Specify the requested supported operation; do not infer a mask.")
        elif operation != "mask" and "scalar" not in params:
            issues.append("Arithmetic operation needs an explicit scalar.")
    elif kind == "ZonalStatistics":
        raster_case = bool(params.get("raster_path") and params.get("zones_path"))
        table_case = bool(
            (params.get("table_path") or single_path)
            and params.get("zone_field")
            and params.get("value_field")
        )
        explicit_pair = (
            isinstance(paths, list)
            and len(paths) == 2
            and all((isinstance(path, str) and path for path in paths))
        )
        if not (raster_case or table_case or explicit_pair):
            issues.append(
                "Supply raster_path + zones_path, or table_path + zone_field + value_field; global statistics are not zonal."
            )
    elif kind == "RenderMap":
        if not (
            params.get("raster_path")
            or params.get("vector_path")
            or (isinstance(paths, list) and paths)
        ):
            issues.append("Rendering needs actual spatial input; a title is insufficient.")
        if params.get("raster_path") and params.get("classification_field"):
            issues.append("classification_field is a vector attribute; omit it for a raster map.")
    return issues
