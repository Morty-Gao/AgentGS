"""Opt-in remote-sensing step contracts; no benchmark or answer dependencies.

The Agent opts raster steps in with _rs_profile. Direct service calls retain
their legacy behavior unless explicitly opted in. Unavailable metadata is
reported as unknown, never invented from a generic band number.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_enabled = ContextVar("rs_contracts_enabled", default=True)
VERSION = "rs-local-v1"
FLAG = "GEOTOOL_RS_LOCAL_CONTRACTS"
PAIR_FIELDS = {
    "RemoteSensingNDVI": (
        ("nir_band_files", "input_nir_paths", "nir_band_file", "input_nir_path"),
        ("red_band_files", "input_red_paths", "red_band_file", "input_red_path"),
        ("nir", "red"),
    ),
    "RemoteSensingNDWI": (
        ("input_swir_paths", "green_band_files", "input_swir_path", "green_band_file"),
        ("input_nir_paths", "nir_band_files", "input_nir_path", "nir_band_file"),
        ("green", "nir"),
    ),
    "RemoteSensingNBR": (
        ("input_nir_paths", "input_nir_path"),
        ("input_swir_paths", "input_swir_path"),
        ("nir", "swir"),
    ),
    "RemoteSensingBatchNDSI": (("green_file_list",), ("swir_file_list",), ("green", "swir")),
    "RemoteSensingCloudMask": (
        ("sr_band_paths", "sr_band_path"),
        ("qa_pixel_paths", "qa_pixel_path"),
        (None, "qa"),
    ),
}
RASTER_STEPS = set(PAIR_FIELDS) | {
    "RemoteSensingNDTI",
    "RemoteSensingTiffDifference",
    "RemoteSensingBatchImageMean",
    "RemoteSensingBatchImageSum",
    "RemoteSensingTiffAverage",
}
SCALAR_CONSUMERS = {
    "RemoteSensingMean",
    "RemoteSensingLinearTrend",
    "RemoteSensingSensSlope",
    "RemoteSensingMannKendallTest",
    "RemoteSensingCoefficientOfVariation",
    "RemoteSensingKurtosis",
    "RemoteSensingMaxValueAndIndex",
    "RemoteSensingMinValueAndIndex",
    "RemoteSensingArgmax",
}
SKILLS = {
    "remote-sensing-data-discovery",
    "remote-sensing-preprocessing",
    "remote-sensing-raster-statistics",
    "remote-sensing-change-detection",
    "remote-sensing-trend-distribution",
}


def enabled() -> bool:
    return _enabled.get()


def skill_profile(skill_id: str) -> dict[str, Any]:
    if not enabled() or skill_id not in SKILLS:
        return {}
    return {
        "version": VERSION,
        "pair_by": ["acquisition_date", "scene", "band_role"],
        "unknown_metadata": "report_unknown; never treat it as verified",
        "date_window": "use rs_date_window=[start,end] only when explicitly specified",
        "batch_cardinality": "one batch OR disjoint per-pair calls, never both",
        "rs_statistic_mode": "per_scene_statistics returns scalars; pixelwise_composite requires aligned grids; legacy retains both outputs",
    }


def _date(text: str) -> str | None:
    match = re.search("(?<!\\d)((?:19|20|21)\\d{2})[-_/]?([01]\\d)[-_/]?([0-3]\\d)(?!\\d)", text)
    if match:
        try:
            return date(*map(int, match.groups())).isoformat()
        except ValueError:
            return None
    match = re.search("(?:^|[_.-])A((?:19|20|21)\\d{2})(\\d{3})(?!\\d)", text, re.I)
    if match:
        year, day = map(int, match.groups())
        result = date(year, 1, 1) + timedelta(days=day - 1)
        if day > 0 and result.year == year:
            return result.isoformat()
    return None


def asset_metadata(path: str, *, inspect_raster: bool = False) -> dict[str, Any]:
    name = Path(str(path)).name
    result: dict[str, Any] = {
        "path": str(path),
        "acquisition_date": _date(name),
        "date_source": "filename",
    }
    band = re.search(
        "(?:^|[_.-])(?:sr_|sur_refl_)?b(?:and)?[_-]?0?(\\d{1,2})(?=[_.-]|$)", name, re.I
    )
    result["band_number"] = int(band[1]) if band else None
    role = re.search("(?:^|[_.-])(qa_pixel|green|red|nir|swir[12]?)(?=[_.-]|$)", name, re.I)
    result["band_role"] = (
        ("qa" if role[1].lower() == "qa_pixel" else re.sub("[12]$", "", role[1].lower()))
        if role
        else None
    )
    if inspect_raster and Path(path).is_file():
        try:
            import rasterio

            with rasterio.open(path) as source:
                tags = {k.casefold(): v for k, v in {**source.tags(), **source.tags(1)}.items()}
                for key in ("acquisition_date", "date_acquired", "sensing_time"):
                    if tags.get(key) and _date(tags[key]):
                        result.update(
                            acquisition_date=_date(tags[key]), date_source="raster_tag:" + key
                        )
                        break
                role = tags.get("band_role")
                if role:
                    role = role.casefold().strip().replace("-", "_").replace(" ", "_")
                    role = {
                        "near_infrared": "nir",
                        "shortwave_infrared": "swir",
                        "swir1": "swir",
                        "swir2": "swir",
                        "qa_pixel": "qa",
                    }.get(role, role)
                    result["band_role"] = (
                        role if role in {"nir", "swir", "green", "red", "qa"} else None
                    )
                for key in ("region", "product", "sensor", "variable", "units"):
                    if tags.get(key):
                        result[key] = tags[key]
        except (OSError, ValueError, ImportError):
            result["metadata_status"] = "unavailable"
    if not result["acquisition_date"]:
        result["date_source"] = "unknown"
    return result


def _values(params: dict, keys: tuple[str, ...]) -> list:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return value if isinstance(value, list) else [value]
    return []


def align_scene_pairs(
    first: list[str], second: list[str], *, reuse_qa: bool = False
) -> list[tuple[str, str]]:
    """Join complete acquisitions, rather than silently dropping missing bands."""
    if not first or not second:
        return []
    if not reuse_qa and len(first) != len(second):
        raise ValueError("unmatched band inventory: paired lists have different lengths")

    def scene_key(path):
        stem = Path(path).stem.casefold()
        stem = re.sub("[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", "", stem)
        stem = re.sub(
            "cloud[_-]?masked|qa[_-]?pixel|(?:sur[_-]?refl[_-]?)?(?:sr[_-]?)?b(?:and)?[_-]?\\d+|green|nir|red|swir[12]?",
            "",
            stem,
        )
        return re.sub("[^a-z0-9]", "", stem)

    pairs, used = ([], set())
    first_meta = [asset_metadata(p, inspect_raster=True) for p in first]
    second_meta = [asset_metadata(p, inspect_raster=True) for p in second]
    for path, meta in zip(first, first_meta):
        candidates = [i for i, other in enumerate(second) if scene_key(path) == scene_key(other)]
        if not candidates and meta["acquisition_date"]:
            candidates = [
                i
                for i, other in enumerate(second_meta)
                if meta["acquisition_date"] == other["acquisition_date"]
            ]
        if len(candidates) != 1:
            raise ValueError(
                "ambiguous or missing acquisition/band partner; provide explicit paired paths"
            )
        index = candidates[0]
        if index in used and (not reuse_qa):
            raise ValueError("one band asset cannot satisfy multiple acquisition pairs")
        pairs.append((path, second[index]))
        used.add(index)
    if len(used) != len(second):
        raise ValueError("unmatched band/QA assets remain; explicitly select the intended cohort")
    return pairs


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


def prepare_step(tool: str, params: dict, context: dict) -> dict:
    """Agent-only opt-in and local fixes. Never touches a non-RS tool."""
    if (
        not enabled()
        or tool not in RASTER_STEPS
        or context.get("remote_sensing_contracts") is False
    ):
        return {}
    params["_rs_profile"] = VERSION
    issues = []
    if (
        tool == "RemoteSensingTiffDifference"
        and isinstance(params.get("image_a_paths"), list)
        and isinstance(params.get("image_b_paths"), list)
    ):
        first, second = (params["image_a_paths"], params["image_b_paths"])
        count = max(1, int(context.get("occurrence_count", 1)))
        index = int(context.get("occurrence_index", 0))
        partition = bool(context.get("rs_auto_consecutive_batch"))
        if len(first) == len(second) and count > 1 and partition:
            if count > len(first) or not 0 <= index < count:
                issues.append(
                    "more repeated difference steps than available pairs; replan instead of replaying"
                )
            else:
                start, end = (len(first) * index // count, len(first) * (index + 1) // count)
                first, second = (first[start:end], second[start:end])
                params["image_a_paths"], params["image_b_paths"] = (first, second)
        params.pop("image_a_path", None)
        params.pop("image_b_path", None)
        if count > 1 and partition and (len(first) == len(second) == 1):
            params["image_a_path"], params["image_b_path"] = (first[0], second[0])
            params.pop("image_a_paths", None)
            params.pop("image_b_paths", None)
    if tool == "RemoteSensingBatchImageMean" and "rs_statistic_mode" not in params:
        chain = list(context.get("tool_chain") or [])
        position = int(context.get("execution_index", 0))
        following = chain[position:] if position else []
        if (
            not context.get("rs_requires_composite")
            and following
            and any((t in SCALAR_CONSUMERS for t in following))
            and all(
                (
                    t in SCALAR_CONSUMERS
                    or t
                    in {
                        "RemoteSensingBatchImageMean",
                        "RemoteSensingDifference",
                        "RemoteSensingDivision",
                        "RemoteSensingPercentageChange",
                        "RemoteSensingKelvinToCelsius",
                    }
                    for t in following
                )
            )
        ):
            params["rs_statistic_mode"] = "per_scene_statistics"
    report = inspect_inputs(tool, params, inspect_raster=True)
    report["issues"].extend(issues)
    params["_rs_binding_issues"] = report["issues"]
    report["rs_statistic_mode"] = params.get("rs_statistic_mode", "legacy")
    return report
