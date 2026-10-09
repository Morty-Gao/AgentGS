"""Inspect raster inputs, distinguishing violations from unknown metadata."""

from pathlib import Path

from .assets import _date, asset_metadata
from .policy import PAIR_FIELDS, VERSION


def _values(params: dict, keys: tuple[str, ...]) -> list:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return value if isinstance(value, list) else [value]
    return []


def inspect_inputs(
    tool: str, params: dict, *, inspect_raster: bool = False, resolve_path=None
) -> dict:
    """Return observable issues, not a claim that the whole task is complete."""
    issues, unknown, assets = ([], [], {})

    def asset(path):
        if not isinstance(path, (str, Path)):
            issues.append("raster paths must be strings")
            return {}
        path = str(path)
        if path not in assets:
            assets[path] = asset_metadata(
                str(resolve_path(path)) if resolve_path else path, inspect_raster=inspect_raster
            )
        return assets[path]

    def check_pair_lists(first, second, roles):
        if not first or len(first) != len(second):
            issues.append("paired raster lists must be non-empty and equal length; no truncation")
            return
        for left, right in zip(first, second):
            a, b = (asset(left), asset(right))
            if str(left) == str(right):
                issues.append("different band roles cannot bind the same raster")
            if a.get("acquisition_date") and b.get("acquisition_date"):
                if a["acquisition_date"] != b["acquisition_date"]:
                    issues.append("paired bands/QA have different acquisition dates")
            else:
                unknown.append("pair acquisition date unavailable")
            for key in ("region", "sensor"):
                if (
                    a.get(key)
                    and b.get(key)
                    and (str(a[key]).strip().casefold() != str(b[key]).strip().casefold())
                ):
                    issues.append("paired rasters have different " + key)
            for item, expected in zip((a, b), roles):
                observed = item.get("band_role")
                if expected and observed and (expected != observed):
                    issues.append(f"band role mismatch: expected {expected}, observed {observed}")
                elif expected and (not observed):
                    unknown.append(
                        "band role unavailable; numeric band is not a sensor-independent role"
                    )
            if (
                all(roles)
                and a.get("band_number") is not None
                and (a.get("band_number") == b.get("band_number"))
            ):
                issues.append("different spectral roles have the same band number")

    if tool in PAIR_FIELDS:
        left, right, roles = PAIR_FIELDS[tool]
        check_pair_lists(_values(params, left), _values(params, right), roles)
    elif tool == "RemoteSensingNDTI":
        pairs = params.get("input_pairs") or []
        check_pair_lists(
            [p.get("input_green_path") for p in pairs],
            [p.get("input_red_path") for p in pairs],
            ("green", "red"),
        )
    else:
        for key in ("file_list", "image_a_paths", "image_b_paths", "image_a_path", "image_b_path"):
            if key in params:
                for path in _values(params, (key,)):
                    asset(path)
    window = params.get("rs_date_window")
    if window is not None:
        if (
            not isinstance(window, list)
            or len(window) != 2
            or (not all((isinstance(v, str) and _date(v) for v in window)))
        ):
            issues.append("rs_date_window must contain two explicit ISO dates")
        elif _date(window[0]) > _date(window[1]):
            issues.append("rs_date_window start is after end")
        else:
            for item in assets.values():
                observed = item.get("acquisition_date")
                if observed and (not _date(window[0]) <= observed <= _date(window[1])):
                    issues.append("acquisition outside declared rs_date_window")
                elif not observed:
                    unknown.append("cannot verify requested date window")
    if tool == "RemoteSensingTiffDifference":
        batch = "image_a_paths" in params or "image_b_paths" in params
        scalar = "image_a_path" in params or "image_b_path" in params
        if batch and scalar:
            issues.append("TiffDifference batch and per-pair arguments are mutually exclusive")
        if batch:
            a, b = (params.get("image_a_paths"), params.get("image_b_paths"))
            if not isinstance(a, list) or not isinstance(b, list) or (not a) or (len(a) != len(b)):
                issues.append("TiffDifference batch arrays must be non-empty and equal length")
    return {
        "version": VERSION,
        "issues": list(dict.fromkeys(issues)),
        "unknown": list(dict.fromkeys(unknown)),
        "assets": list(assets.values()),
    }
