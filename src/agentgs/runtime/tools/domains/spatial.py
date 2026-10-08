"""spatial scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from agentgs.runtime.tools.context import paths as task_paths
from agentgs.runtime.tools.domains.shared import (
    _bbox_area,
    _clean,
    _haversine,
    _numeric_input,
    _point_pair,
    _read_csv,
    _result,
)
from agentgs.runtime.tools.resources import _resolve_path


def _decimal_to_dms(params: dict[str, Any]) -> str:
    text = _numeric_input(params)
    parts = [part.strip() for part in text.split(",")]
    value = float(parts[0])
    axis = parts[1].lower() if len(parts) > 1 else ""
    absolute = abs(value)
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    hemisphere = (
        "S"
        if axis == "lat" and value < 0
        else "N"
        if axis == "lat"
        else "W"
        if axis == "lon" and value < 0
        else "E"
        if axis == "lon"
        else "-"
        if value < 0
        else "+"
    )
    return _result(
        dms=f"""{degrees}°{minutes}'{seconds:.2f}" {hemisphere}""", decimal_degrees=value, axis=axis
    )


def _distance(params: dict[str, Any]) -> str:
    lat1, lon1, lat2, lon2 = _point_pair(params)
    return _result(distance_km=_haversine(lat1, lon1, lat2, lon2))


def _bearing(params: dict[str, Any]) -> str:
    lat1, lon1, lat2, lon2 = _point_pair(params)
    phi1, phi2 = (math.radians(lat1), math.radians(lat2))
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return _result(bearing_degrees=(math.degrees(math.atan2(y, x)) + 360) % 360)


def _bbox_tool(params: dict[str, Any]) -> str:
    bbox = params.get("bbox")
    if bbox is None:
        bbox = [float(item.strip()) for item in _numeric_input(params).split(",")]
    return _result(bbox=bbox, area_km2=_bbox_area(bbox))


def _crs_check(params: dict[str, Any]) -> str:
    values = params.get("crs_values")
    if not values:
        try:
            path = next(
                (path for path in _extract_paths(params) if path.suffix.casefold() == ".csv")
            )
            table = _read_csv(path)
            crs_field = next(
                (field for field in table.columns if "crs" in str(field).casefold()), None
            )
            if crs_field:
                values = table[crs_field].dropna().astype(str).tolist()
        except (FileNotFoundError, StopIteration):
            values = None
    if not values:
        values = re.split("[,;]", _numeric_input(params))
    normalized = [
        str(value).strip().upper().replace(" ", "") for value in values if str(value).strip()
    ]
    return _result(values=normalized, consistent=len(set(normalized)) <= 1)


def _iter_path_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).casefold()
            if key_lower in {
                "paths",
                "path",
                "input_path",
                "file_path",
                "data",
                "source",
                "input_assets",
                "source_assets",
                "files",
                "file_list",
                "output_path",
                "standardized_points",
                "standardized_layer",
                "clean_points",
                "clean_leak_points",
                "buffer_output",
                "overlay_output",
                "raster_output",
                "binary_map_path",
                "change_image_path",
                "filtered_records_path",
            } or any((token in key_lower for token in ("_path", "_file", "_raster", "_layer"))):
                yield from _iter_path_values(item)
            elif isinstance(item, (dict, list, tuple)):
                yield from _iter_path_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_path_values(item)
    elif isinstance(value, (str, Path)):
        text = str(value)
        if re.search("\\.(?:csv|geojson|json|jsonl|tif|tiff|npy)$", text, flags=re.IGNORECASE):
            yield value


def _extract_paths(params: dict[str, Any]) -> list[Path]:
    raw_values = list(_iter_path_values(params))
    if not raw_values and params.get("input_text"):
        raw_values = re.findall(
            "(?:[A-Za-z]:)?[^;\\uff1b,\\uff0c\\n={}]+\\.(?:csv|geojson|json|jsonl|tif|tiff|npy)",
            str(params["input_text"]),
            flags=re.IGNORECASE,
        )
    paths: list[Path] = []
    for value in raw_values:
        try:
            resolved = _resolve_path(value)
            if resolved.is_dir():
                paths.extend(
                    (path.resolve() for path in sorted(resolved.iterdir()) if path.is_file())
                )
            elif resolved not in paths:
                paths.append(resolved)
        except FileNotFoundError:
            continue
    if not paths:
        raise FileNotFoundError(
            "No concrete input assets could be resolved from the tool arguments"
        )
    return paths


def _feature_collection_from_csv(path: Path, params: dict[str, Any]) -> dict[str, Any]:
    table = _read_csv(path)
    lon_field = str(
        params.get("lon_field")
        or next(
            (
                name
                for name in ("lon", "long", "longitude", "longitude_wgs84")
                if name in table.columns
            ),
            "",
        )
    )
    lat_field = str(
        params.get("lat_field")
        or next(
            (name for name in ("lat", "latitude", "latitude_wgs84") if name in table.columns), ""
        )
    )
    if not lon_field or not lat_field:
        raise ValueError(f"Could not identify longitude/latitude columns in {path}")
    features = []
    for _, row in table.iterrows():
        try:
            lon, lat = (float(row[lon_field]), float(row[lat_field]))
        except (TypeError, ValueError):
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        properties = {
            str(key): _clean(value)
            for key, value in row.to_dict().items()
            if key not in (lon_field, lat_field)
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
    }


def _standardize_spatial(params: dict[str, Any]) -> str:
    outputs = []
    skipped = []
    for path in _extract_paths(params):
        try:
            output = task_paths().outputs / "vector" / f"{path.stem}_standardized.geojson"
            output.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".csv":
                collection = _feature_collection_from_csv(path, params)
            elif path.suffix.lower() in (".geojson", ".json"):
                collection = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ValueError("not a supported vector input")
            output.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
            outputs.append(
                {
                    "source": str(path),
                    "output": str(output),
                    "feature_count": len(collection.get("features", [])),
                    "crs": "EPSG:4326",
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"source": str(path), "reason": str(exc)})
    if not outputs:
        raise ValueError(f"No input layer could be standardized: {skipped}")
    return _result(standardized_layers=outputs, skipped_layers=skipped)


def _validate_spatial(params: dict[str, Any]) -> str:
    reports = []
    skipped = []
    for path in _extract_paths(params):
        try:
            if path.suffix.lower() == ".csv":
                collection = _feature_collection_from_csv(path, params)
            elif path.suffix.lower() in (".geojson", ".json"):
                collection = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ValueError("not a supported vector input")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append({"source": str(path), "reason": str(exc)})
            continue
        valid, invalid, duplicates, seen = ([], 0, 0, set())
        for feature in collection.get("features", []):
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates") if geometry.get("type") == "Point" else None
            if not coords or not (
                -180 <= float(coords[0]) <= 180 and -90 <= float(coords[1]) <= 90
            ):
                invalid += 1
                continue
            key = (round(float(coords[0]), 10), round(float(coords[1]), 10))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            valid.append(feature)
        output = task_paths().outputs / "vector" / f"{path.stem}_validated.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"type": "FeatureCollection", "features": valid}, ensure_ascii=False),
            encoding="utf-8",
        )
        reports.append(
            {
                "source": str(path),
                "clean_output": str(output),
                "valid": len(valid),
                "invalid": invalid,
                "duplicates_removed": duplicates,
            }
        )
    if not reports:
        raise ValueError(f"No input layer could be validated: {skipped}")
    return _result(validation_reports=reports, skipped_layers=skipped)


def _load_points(path: Path) -> list[tuple[float, float, dict[str, Any]]]:
    collection = (
        _feature_collection_from_csv(path, {})
        if path.suffix.lower() == ".csv"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    points = []
    for feature in collection.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "Point":
            lon, lat = geometry["coordinates"][:2]
            points.append((float(lat), float(lon), feature.get("properties") or {}))
    return points


def _nearest_neighbor(params: dict[str, Any]) -> str:
    points = _load_points(_extract_paths(params)[0])
    k = max(1, int(params.get("k") or params.get("neighbor_count") or 1))
    neighbours = []
    for index, first in enumerate(points):
        distances = [
            (_haversine(first[0], first[1], second[0], second[1]), other)
            for other, second in enumerate(points)
            if other != index
        ]
        for rank, (distance, other) in enumerate(sorted(distances)[:k], 1):
            neighbours.append(
                {
                    "source_index": index,
                    "neighbor_index": other,
                    "neighbor_rank": rank,
                    "distance_km": distance,
                }
            )
    return _result(neighbor_count=k, nearest_neighbours=neighbours)


def _buffer_points(params: dict[str, Any]) -> str:
    path = _extract_paths(params)[0]
    requested_distances = params.get("buffer_distances_km")
    if isinstance(requested_distances, (list, tuple)) and requested_distances:
        distances = [float(value) for value in requested_distances]
    else:
        distances = [float(params.get("distance_km") or params.get("buffer_km") or 50)]
    features = []
    for distance in distances:
        for lat, lon, properties in _load_points(path):
            dlat = distance / 111.32
            dlon = distance / max(111.32 * math.cos(math.radians(lat)), 1e-06)
            ring = [
                [lon - dlon, lat - dlat],
                [lon + dlon, lat - dlat],
                [lon + dlon, lat + dlat],
                [lon - dlon, lat + dlat],
                [lon - dlon, lat - dlat],
            ]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {**properties, "buffer_km": distance},
                }
            )
    distance_label = "_".join((f"{value:g}" for value in distances))
    output = task_paths().outputs / "vector" / f"{path.stem}_buffer_{distance_label}km.geojson"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    return _result(
        buffer_output=output,
        buffer_distances_km=distances,
        feature_count=len(features),
        method="spherical point-buffer envelope",
    )


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi or 1e-15) + xi:
            inside = not inside
        j = i
    return inside


def _overlay(params: dict[str, Any]) -> str:
    paths = _extract_paths(params)
    if len(paths) < 2:
        return _result(
            operation="overlay",
            inspected_paths=[str(path) for path in paths],
            status="one input resolved; no second layer supplied",
        )

    def has_points(path: Path) -> bool:
        try:
            return bool(_load_points(path))
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    point_path = next(
        (
            path
            for path in paths
            if path.suffix.casefold() in (".csv", ".json", ".geojson") and has_points(path)
        ),
        None,
    )
    polygon_paths: list[Path] = []
    for path in paths:
        if path.suffix.casefold() not in (".json", ".geojson"):
            continue
        try:
            collection = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        features = collection.get("features", []) if isinstance(collection, dict) else []
        if any(
            (
                (feature.get("geometry") or {}).get("type") in ("Polygon", "MultiPolygon")
                for feature in features
            )
        ):
            polygon_paths.append(path)
    polygon_path = polygon_paths[-1] if polygon_paths else None
    if point_path is None or polygon_path is None:
        if len(polygon_paths) >= 2:
            output = (
                task_paths().outputs / "vector" / f"{polygon_paths[0].stem}_polygon_overlay.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "operation": "polygon_overlay_inspection",
                "layers": [str(path) for path in polygon_paths],
                "status": "polygon layers resolved and inspected",
            }
            output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return _result(overlay_output=output, joined_count=0, **payload)
        return _result(
            operation="overlay",
            inspected_paths=[str(path) for path in paths],
            status="inputs resolved; no compatible point/polygon pair",
        )
    polygons = json.loads(polygon_path.read_text(encoding="utf-8")).get("features", [])
    joined = []
    for lat, lon, properties in _load_points(point_path):
        for feature in polygons:
            geometry = feature.get("geometry") or {}
            rings = geometry.get("coordinates", [])
            if geometry.get("type") == "MultiPolygon":
                rings = [polygon[0] for polygon in rings]
            elif geometry.get("type") == "Polygon":
                rings = rings[:1]
            else:
                continue
            if any((_point_in_ring(lon, lat, ring) for ring in rings)):
                joined.append(
                    {
                        **properties,
                        **(feature.get("properties") or {}),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )
                break
    output = task_paths().outputs / "vector" / f"{point_path.stem}_overlay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(joined, ensure_ascii=False), encoding="utf-8")
    return _result(overlay_output=output, joined_count=len(joined))


def _batch_vector(params: dict[str, Any]) -> str:
    return _standardize_spatial(params)


def _interpolate_idw(params: dict[str, Any]) -> str:
    from agentgs.runtime.tools.scientific.selection import (
        explicit_spatial_paths,
        select_observation_rows,
    )
    from agentgs.runtime.tools.scientific.spatial import interpolate_idw

    paths = explicit_spatial_paths(params, "idw", _resolve_path, _extract_paths)
    selected, selection = select_observation_rows(params, paths, task_paths().outputs)
    result = interpolate_idw(params, selected, task_paths().outputs)
    if selection:
        result["observation_selection"] = selection
    return _result(**result)


def _raster_algebra(params: dict[str, Any]) -> str:
    from agentgs.runtime.tools.scientific.selection import explicit_spatial_paths
    from agentgs.runtime.tools.scientific.spatial import raster_algebra

    return _result(
        **raster_algebra(
            params,
            explicit_spatial_paths(params, "raster", _resolve_path, _extract_paths),
            task_paths().outputs,
        )
    )


def _zonal_stats(params: dict[str, Any]) -> str:
    from agentgs.runtime.tools.scientific.selection import explicit_spatial_paths
    from agentgs.runtime.tools.scientific.spatial import zonal_statistics

    return _result(
        **zonal_statistics(
            params,
            explicit_spatial_paths(params, "zonal", _resolve_path, _extract_paths),
            task_paths().outputs,
        )
    )


def _render_map(params: dict[str, Any]) -> str:
    from agentgs.runtime.tools.scientific.selection import explicit_spatial_paths
    from agentgs.runtime.tools.scientific.spatial import render_spatial_map

    return _result(
        **render_spatial_map(
            params,
            explicit_spatial_paths(params, "render", _resolve_path, _extract_paths),
            task_paths().outputs,
        )
    )
