"""Public argument parsing and normalization shared by binding and dispatch."""

import json
from typing import Any

__all__ = ["normalize_arguments", "parse_arguments"]

_ALIASES = {
    "ssp": "scenario",
    "ssp_scenario": "scenario",
    "ssp_horizon": "horizon_year",
    "year": "horizon_year",
    "start": "year_start",
    "end": "year_end",
    "grouping_fields": "group_by",
    "grouping_field": "group_by",
    "expocode": "glodap_expocode",
    "glodap_expocode": "expocode",
    "station_id": "station",
    "cruise_id": "cruise",
    "fco2rec_flag": "fCO2rec_flag",
    "longitude": "lon",
    "latitude": "lat",
}


def _flatten_parameter_wrappers(value: dict[str, Any]) -> dict[str, Any]:
    """Expose authoritative locator fields without discarding their provenance."""
    flattened = dict(value)
    for key in ("source_locator", "arguments", "parameters", "params", "locator", "context"):
        nested = value.get(key)
        if isinstance(nested, dict):
            for nested_key, nested_value in nested.items():
                current = flattened.get(nested_key)
                if (
                    nested_key not in flattened
                    or current is None
                    or current == ""
                    or (current == "*")
                ):
                    flattened[nested_key] = nested_value
    nested_input = value.get("input")
    if isinstance(nested_input, dict):
        for nested_key, nested_value in nested_input.items():
            flattened.setdefault(nested_key, nested_value)
    elif isinstance(nested_input, str) and nested_input.strip().startswith(("{", "[")):
        try:
            decoded = json.loads(nested_input)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            for nested_key, nested_value in decoded.items():
                flattened.setdefault(nested_key, nested_value)
    return flattened


def normalize_arguments(value: dict[str, Any]) -> dict[str, Any]:
    """Flatten wrappers and apply aliases, preserving authoritative values."""
    params = _flatten_parameter_wrappers(value)
    for source, target in _ALIASES.items():
        if target not in params or params.get(target) in (None, "", []):
            if params.get(source) not in (None, "", []):
                params[target] = params[source]
    if isinstance(params.get("input"), dict):
        params.update({k: v for k, v in params["input"].items() if params.get(k) in (None, "", [])})
    return params


def parse_arguments(input_str: str) -> dict[str, Any]:
    """Decode JSON or retain plain text using the existing invocation forms."""
    text = (input_str or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"input_text": text}
    if isinstance(value, dict):
        return normalize_arguments(value)
    if isinstance(value, list):
        return {"x": value}
    return {"value": value}


# Compatibility for callers of the earlier private entry point.
_params = parse_arguments
