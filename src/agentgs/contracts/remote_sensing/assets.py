"""Observable raster metadata and acquisition-based scene pairing."""

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any


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
