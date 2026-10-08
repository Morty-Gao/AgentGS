"""Preserved scientific spatial operations from the pre-rollback source.
Only the IDW/raster/zonal/render dependency closure is retained here.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path
from uuid import uuid4


class UnsupportedOperation(ValueError):
    pass


class ScientificResultError(ValueError):
    pass


def coordinate_pair(values):
    if len(values) != 4:
        raise ValueError("Expected exactly lat1, lon1, lat2, lon2")
    values = tuple((float(value) for value in values))
    if any((not math.isfinite(value) for value in values)):
        raise ValueError("Coordinates must be finite")
    if any((abs(values[index]) > (90 if index % 2 == 0 else 180) for index in range(4))):
        raise ValueError("Latitude must be [-90,90] and longitude [-180,180]")
    return values


def spherical_distance(*values):
    lat1, lon1, lat2, lon2 = coordinate_pair(values)
    phi1, phi2 = (math.radians(lat1), math.radians(lat2))
    a = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    a = min(1.0, max(0.0, a))
    return 6371.0088 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def spherical_bbox_area(values):
    if len(values) != 4:
        raise ValueError("bbox requires min_lon,min_lat,max_lon,max_lat")
    west, south, east, north = map(float, values)
    coordinate_pair((south, west, north, east))
    if not west < east or not south < north:
        raise ValueError(
            "bbox requires increasing bounds; split antimeridian-crossing boxes explicitly"
        )
    return (
        6371.0088**2
        * math.radians(east - west)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )


def read_vector(path, params=None, *, allow_invalid=False):
    """Read source coordinates without relabelling a projected layer as WGS84."""
    from rasterio.crs import CRS
    from rasterio.warp import transform_geom

    params = params or {}
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            lon_field = params.get("lon_field") or next(
                (key for key in ("lon", "long", "longitude", "longitude_wgs84") if key in fields),
                None,
            )
            lat_field = params.get("lat_field") or next(
                (key for key in ("lat", "latitude", "latitude_wgs84") if key in fields), None
            )
            if lon_field not in fields or lat_field not in fields:
                raise ValueError("Point CSV requires named coordinate columns")
            features = []
            for index, row in enumerate(reader):
                try:
                    coordinates = [float(row[lon_field]), float(row[lat_field])]
                except (TypeError, ValueError):
                    coordinates = None
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": coordinates}
                        if coordinates is not None
                        else None,
                        "properties": {
                            key: value
                            for key, value in row.items()
                            if key not in (lon_field, lat_field)
                        },
                    }
                )
        source_crs = params.get("source_crs", "EPSG:4326")
        value = {"type": "FeatureCollection", "features": features}
    elif path.suffix.casefold() in {".geojson", ".json"}:
        value = json.loads(path.read_text(encoding="utf-8"))
        declared = value.get("crs", {}).get("properties", {}).get("name")
        source_crs = params.get("source_crs") or declared or "EPSG:4326"
        if (
            declared
            and params.get("source_crs")
            and (CRS.from_user_input(declared) != CRS.from_user_input(source_crs))
        ):
            raise ValueError("source_crs contradicts the layer CRS")
    else:
        raise UnsupportedOperation("Vector input must be CSV or GeoJSON")
    if (
        value.get("type") != "FeatureCollection"
        or not isinstance(value.get("features"), list)
        or (not value["features"])
    ):
        raise ValueError("Expected a nonempty FeatureCollection")
    source = CRS.from_user_input(source_crs)
    if not source:
        raise ValueError("A valid source CRS is required")
    transformed = []
    for feature in value["features"]:
        try:
            if feature.get("type") != "Feature" or not isinstance(feature.get("geometry"), dict):
                raise ValueError("Missing feature geometry")
            geometry = feature["geometry"]
            if source != CRS.from_epsg(4326):
                geometry = transform_geom(source, "EPSG:4326", geometry)
            _valid_geometry(geometry)
            transformed.append({**feature, "geometry": geometry})
        except (TypeError, ValueError, KeyError) as exc:
            if not allow_invalid:
                raise ValueError(f"Invalid geometry in {path.name}: {exc}") from exc
            transformed.append({**feature, "geometry": None, "_validation_error": str(exc)})
    return (
        {
            "type": "FeatureCollection",
            "features": transformed,
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        },
        str(source),
    )


def _valid_geometry(geometry):
    from shapely.geometry import shape
    from shapely.validation import explain_validity

    if not geometry or geometry.get("type") not in {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }:
        raise ValueError("Unsupported or empty geometry")
    coordinates = list(_coordinates(geometry.get("coordinates", [])))
    if not coordinates or any(
        (
            not math.isfinite(lon) or not math.isfinite(lat) or abs(lon) > 180 or (abs(lat) > 90)
            for lon, lat in coordinates
        )
    ):
        raise ValueError("Invalid longitude/latitude coordinate")
    result = shape(geometry)
    if result.is_empty or not result.is_valid:
        raise ValueError(f"Invalid geometry: {explain_validity(result)}")
    return result


def _collection(path: Path):
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        features = []
        for row in rows:
            lon = next(
                (row[key] for key in ("lon", "long", "longitude", "longitude_wgs84") if key in row),
                None,
            )
            lat = next(
                (row[key] for key in ("lat", "latitude", "latitude_wgs84") if key in row), None
            )
            if lon is None or lat is None:
                raise UnsupportedOperation(
                    "Point CSV requires explicit longitude and latitude columns"
                )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": row,
                }
            )
        value = {"type": "FeatureCollection", "features": features}
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise UnsupportedOperation("Expected a GeoJSON FeatureCollection")
    crs = value.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if str(crs).upper() not in {"EPSG:4326", "URN:OGC:DEF:CRS:OGC:1.3:CRS84", "CRS84"}:
        raise UnsupportedOperation(
            "This vector operation requires WGS84 longitude/latitude GeoJSON"
        )
    for feature in value["features"]:
        _valid_geometry(feature.get("geometry"))
        for lon, lat in _coordinates((feature.get("geometry") or {}).get("coordinates", [])):
            if (
                not math.isfinite(lon)
                or not math.isfinite(lat)
                or (not -180 <= lon <= 180)
                or (not -90 <= lat <= 90)
            ):
                raise ValueError("Invalid longitude/latitude coordinate")
    return value


def _coordinates(value):
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and all((isinstance(item, (int, float)) for item in value[:2]))
    ):
        yield (float(value[0]), float(value[1]))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _coordinates(child)


def _unique_output(path):
    return path.with_name(f"{path.stem}_{uuid4().hex}{path.suffix}")


def interpolate_idw(params, paths, output_root):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    if len(paths) != 1:
        raise UnsupportedOperation("IDW requires exactly one measured point layer")
    layer, _ = read_vector(paths[0], params)
    field = params.get("value_field") or params.get("field")
    if not isinstance(field, str) or not field:
        raise ValueError(
            "IDW requires explicit value_field; identifiers are never selected as measurements"
        )
    locations = {}
    for feature in layer["features"]:
        if feature["geometry"]["type"] != "Point":
            raise ValueError("IDW requires Point observations")
        coordinate = tuple(feature["geometry"]["coordinates"][:2])
        value = float(feature.get("properties", {}).get(field))
        if not math.isfinite(value):
            raise ValueError("IDW observations must be finite")
        locations.setdefault(coordinate, []).append(value)
    if len(locations) < 2:
        raise ValueError("IDW requires at least two distinct sample locations")
    size, power = (params.get("grid_size", 50), float(params.get("power", 2)))
    if (
        type(size) is not int
        or not 2 <= size <= 250
        or (not math.isfinite(power))
        or (not 0 < power <= 10)
    ):
        raise ValueError("Use grid_size 2..250 and finite power in (0,10]")
    xs, ys = zip(*locations)
    bbox = list(map(float, params.get("bbox", [min(xs), min(ys), max(xs), max(ys)])))
    spherical_bbox_area(bbox)
    if bbox[2] - bbox[0] > 180:
        raise UnsupportedOperation("Split antimeridian-spanning interpolation grids")
    transform = from_bounds(*bbox, size, size)
    gx, gy = np.meshgrid(
        bbox[0] + (np.arange(size) + 0.5) * (bbox[2] - bbox[0]) / size,
        bbox[3] - (np.arange(size) + 0.5) * (bbox[3] - bbox[1]) / size,
    )
    numerator, denominator = (np.zeros_like(gx), np.zeros_like(gx))
    exact = np.full_like(gx, np.nan)
    phi = np.deg2rad(gy)
    for (lon, lat), values in locations.items():
        mean = sum(values) / len(values)
        a = (
            np.sin((phi - math.radians(lat)) / 2) ** 2
            + np.cos(phi) * math.cos(math.radians(lat)) * np.sin(np.deg2rad(gx - lon) / 2) ** 2
        )
        distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        weight = 1 / np.maximum(distance, 1e-09) ** power
        numerator += weight * mean
        denominator += weight
        exact[distance <= 1e-09] = mean
    grid = np.where(np.isfinite(exact), exact, numerator / denominator)
    if not np.isfinite(grid).all():
        raise ScientificResultError("IDW produced nonfinite values")
    safe_field = re.sub("[^A-Za-z0-9_-]", "_", field)
    output = _unique_output(output_root / "raster" / f"{paths[0].stem}_{safe_field}_idw.tif")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="float64",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dataset:
        dataset.write(grid, 1)
        dataset.update_tags(method="spherical-distance IDW", variable=field, power=str(power))
    return {
        "raster_output": str(output),
        "value_field": field,
        "shape": [size, size],
        "bbox": bbox,
        "crs": "EPSG:4326",
        "method": "inverse spherical-distance weighting at pixel centers",
        "power": power,
        "observation_count": len(layer["features"]),
        "distinct_locations": len(locations),
        "duplicate_location_rule": "arithmetic mean",
        "finite_pixels": int(grid.size),
    }


def raster_algebra(params, paths, output_root):
    import numpy as np
    import rasterio

    from agentgs.runtime.tools.scientific.raster import read_scientific_raster

    if len(paths) != 1 or paths[0].suffix.casefold() not in {".npy", ".tif", ".tiff"}:
        raise UnsupportedOperation("Raster algebra requires exactly one NPY or GeoTIFF")
    if params.get("expression"):
        raise UnsupportedOperation(
            "Free-form expressions are unsupported; specify operation and scalar explicitly"
        )
    operation = str(params.get("operation", "mask")).casefold()
    if operation not in {"mask", "add", "subtract", "multiply", "divide"}:
        raise UnsupportedOperation(
            "Supported raster operations: mask, add, subtract, multiply, divide"
        )
    profile = None
    if paths[0].suffix.casefold() == ".npy":
        array = np.load(paths[0], allow_pickle=False).astype("float64")
    else:
        with rasterio.open(paths[0]) as dataset:
            band = params.get("band", 1 if dataset.count == 1 else None)
            if type(band) is not int or not 1 <= band <= dataset.count:
                raise ValueError("Multiband raster algebra requires an explicit valid band")
            if dataset.crs is None:
                raise ValueError("GeoTIFF requires a declared CRS")
            array, _ = read_scientific_raster(paths[0], band=band)
            profile = dataset.profile.copy()
    if array.ndim != 2 or not array.size or (not np.isfinite(array).any()):
        raise ValueError("Raster must be a nonempty 2D array with finite observations")
    if operation == "mask":
        lower, upper = (float(params.get("minimum", 0)), float(params.get("maximum", math.inf)))
        if not math.isfinite(lower) or math.isnan(upper) or upper < lower:
            raise ValueError("Mask bounds must be ordered and minimum must be finite")
        result = np.where(np.isfinite(array) & (array >= lower) & (array <= upper), array, np.nan)
    else:
        scalar = float(params["scalar"])
        if not math.isfinite(scalar) or (operation == "divide" and scalar == 0):
            raise ValueError("Scalar must be finite and division by zero is undefined")
        result = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
        }[operation](array, scalar)
        if np.any(np.isfinite(array) & ~np.isfinite(result)):
            raise ScientificResultError("Arithmetic overflow")
    if not np.isfinite(result).any():
        raise ScientificResultError("The requested operation leaves no finite raster cells")
    output = _unique_output(
        output_root / "raster" / (paths[0].stem + "_algebra" + (".tif" if profile else ".npy"))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if profile:
        profile.update(dtype="float64", count=1, nodata=np.nan)
        with rasterio.open(output, "w", **profile) as dataset:
            dataset.write(result, 1)
            dataset.update_tags(operation=operation)
    else:
        np.save(output, result)
    return {
        "raster_output": str(output),
        "finite_pixels": int(np.isfinite(result).sum()),
        "operation": operation,
        "crs": str(profile["crs"]) if profile else None,
        "georeferenced": bool(profile),
    }


def zonal_statistics(params, paths, output_root):
    import numpy as np

    from agentgs.runtime.tools.scientific.raster import read_scientific_raster

    operation = str(params.get("operation", params.get("statistic", "mean"))).casefold()
    if operation not in {"mean", "sum", "minimum", "maximum", "count"}:
        raise UnsupportedOperation(
            "Supported zonal operations are mean, sum, minimum, maximum and count"
        )
    raster_paths = [path for path in paths if path.suffix.casefold() in {".npy", ".tif", ".tiff"}]
    if raster_paths:
        if len(raster_paths) != 1 or raster_paths[0].suffix.casefold() == ".npy":
            raise UnsupportedOperation(
                "Zonal raster statistics require one georeferenced GeoTIFF; raw NPY has no spatial transform"
            )
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.warp import transform_geom

        zone_paths = [path for path in paths if path.suffix.casefold() in {".json", ".geojson"}]
        if len(zone_paths) != 1:
            raise UnsupportedOperation(
                "Zonal statistics require an explicit polygon zone layer; global statistics are not zonal statistics"
            )
        if len(paths) != 2:
            raise UnsupportedOperation(
                "Zonal statistics require exactly one raster and one zone layer; extra sources are ambiguous"
            )
        for key, expected in (("raster_path", raster_paths[0]), ("zones_path", zone_paths[0])):
            if (
                params.get(key) is not None
                and Path(params[key]).expanduser().resolve() != expected.resolve()
            ):
                raise ValueError(f"{key} does not identify the declared input role")
        if params.get("zones_crs") and str(params["zones_crs"]).upper() not in {
            "EPSG:4326",
            "CRS84",
        }:
            raise UnsupportedOperation("zones_crs must agree with WGS84 GeoJSON coordinates")
        zones = _collection(zone_paths[0])["features"]
        if not zones or any(
            (row["geometry"]["type"] not in {"Polygon", "MultiPolygon"} for row in zones)
        ):
            raise UnsupportedOperation("Zone layer must contain Polygon/MultiPolygon features")
        zone_field = params.get("zone_field", "zone_id")
        zone_ids = [(zone.get("properties") or {}).get(zone_field) for zone in zones]
        if any(
            (
                value is None or isinstance(value, (dict, list, bool)) or str(value).strip() == ""
                for value in zone_ids
            )
        ) or len(set(map(str, zone_ids))) != len(zone_ids):
            raise ValueError("Every zone requires a unique nonempty identifier in zone_field")
        summaries = []
        with rasterio.open(raster_paths[0]) as dataset:
            if dataset.crs is None:
                raise UnsupportedOperation("Raster CRS is required for zonal statistics")
            if (
                params.get("raster_crs")
                and rasterio.crs.CRS.from_user_input(params["raster_crs"]) != dataset.crs
            ):
                raise ValueError("raster_crs does not match the selected raster")
            band = params.get("band")
            if band is None:
                if dataset.count != 1:
                    raise UnsupportedOperation(
                        "Multiband zonal statistics require an explicit band"
                    )
                band = 1
            if (
                isinstance(band, bool)
                or not isinstance(band, int)
                or (not 1 <= band <= dataset.count)
            ):
                raise ValueError("band must identify an existing raster band")
            tags = {**dataset.tags(), **dataset.tags(band)}
            for key in ("variable", "time", "year", "region"):
                if params.get(key) is not None and str(tags.get(key)) != str(params[key]):
                    raise ValueError(f"Selected raster has no matching {key} metadata")
            array, interpretation = read_scientific_raster(raster_paths[0], band=band)
            for index, zone in enumerate(zones):
                geometry = transform_geom("EPSG:4326", dataset.crs, zone["geometry"])
                mask = geometry_mask(
                    [geometry],
                    out_shape=array.shape,
                    transform=dataset.transform,
                    invert=True,
                    all_touched=False,
                )
                values = array[mask]
                values = values[np.isfinite(values)]
                if not values.size:
                    raise ValueError(f"Zone {index} has no finite covered raster cells")
                summaries.append(
                    {
                        "zone_id": zone_ids[index],
                        "count": int(values.size),
                        "mean": float(values.mean()),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "sum": float(values.sum()),
                    }
                )
            raster_crs = str(dataset.crs)
        for summary in summaries:
            summary["statistic"] = operation
            summary["value"] = summary[operation]
        return {
            "source": str(raster_paths[0]),
            "zones_source": str(zone_paths[0]),
            "zone_statistics": summaries,
            "zone_count": len(summaries),
            "method": "pixel-center polygon mask",
            "area_weighted": False,
            "raster_crs": raster_crs,
            "zones_crs": "EPSG:4326",
            "band": band,
            "operation": operation,
            "zone_field": zone_field,
            "selected_metadata": tags,
            "input_interpretation": interpretation,
        }
    if (
        len(paths) != 1
        or paths[0].suffix.casefold() != ".csv"
        or (not params.get("zone_field"))
        or (not params.get("value_field"))
    ):
        raise UnsupportedOperation(
            "Tabular zonal statistics require one CSV and explicit zone_field/value_field"
        )
    groups = {}
    with paths[0].open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            zone, value = (row[params["zone_field"]], float(row[params["value_field"]]))
            if not zone or not math.isfinite(value):
                raise ValueError("Zone identifiers and finite measured values are required")
            groups.setdefault(zone, []).append(value)
    if not groups:
        raise ValueError("No observations available for zonal statistics")
    summaries = [
        {
            "zone_id": zone,
            "count": len(values),
            "mean": sum(values) / len(values),
            "sum": sum(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for zone, values in groups.items()
    ]
    for summary in summaries:
        summary.update(statistic=operation, value=summary[operation])
    return {
        "zone_statistics": summaries,
        "zone_count": len(groups),
        "operation": operation,
        "source": str(paths[0]),
        "zone_field": params["zone_field"],
        "value_field": params["value_field"],
        "method": "explicit table group aggregation",
    }


def render_spatial_map(params, paths, output_root):
    import numpy as np
    from PIL import Image, ImageDraw

    from agentgs.runtime.tools.scientific.raster import read_scientific_raster

    if not paths:
        raise UnsupportedOperation("Map rendering requires actual spatial input")
    title = str(params.get("title", "Spatial data map"))
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    rasters = [path for path in paths if path.suffix.casefold() in {".npy", ".tif", ".tiff"}]
    feature_count = 0
    if rasters:
        if len(paths) != 1 or rasters[0].suffix.casefold() == ".npy":
            raise UnsupportedOperation(
                "Raster rendering supports one georeferenced GeoTIFF; NPY or mixed layers need explicit spatial alignment"
            )
        import rasterio

        with rasterio.open(rasters[0]) as dataset:
            if dataset.crs is None:
                raise UnsupportedOperation("A raster map requires an explicit CRS")
            if dataset.transform.b != 0 or dataset.transform.d != 0:
                raise UnsupportedOperation(
                    "Rotated raster grids require reprojection before rendering"
                )
            if dataset.transform.a <= 0 or dataset.transform.e >= 0:
                raise UnsupportedOperation("Rendering requires a north-up, east-right raster grid")
            if params.get("classification_field"):
                raise UnsupportedOperation(
                    "classification_field applies to vector attributes; raster maps use the numeric value ramp"
                )
            band = params.get("band", 1 if dataset.count == 1 else None)
            if type(band) is not int or not 1 <= band <= dataset.count:
                raise ValueError("Multiband rendering requires an explicit valid band")
            values, interpretation = read_scientific_raster(rasters[0], band=band)
            valid = np.isfinite(values)
            if not valid.any():
                raise ValueError("Cannot render a raster with no finite observations")
            low, high = (float(values[valid].min()), float(values[valid].max()))
            normalized = np.zeros_like(values)
            normalized[valid] = (values[valid] - low) / (high - low) if high != low else 0.5
            rgb = np.full((*values.shape, 3), 255, dtype="uint8")
            rgb[valid, 0] = np.asarray(normalized[valid] * 230, dtype="uint8")
            rgb[valid, 1] = 80
            rgb[valid, 2] = np.asarray((1 - normalized[valid]) * 230, dtype="uint8")
            patch = Image.fromarray(rgb)
            patch.thumbnail((880, 530))
            image.paste(patch, (60, 80))
            extent, crs = (list(dataset.bounds), str(dataset.crs))
            render_info = {
                "finite_pixels": int(valid.sum()),
                "value_range": [low, high],
                "band": band,
                "input_interpretation": interpretation,
            }
            from pyproj import CRS

            projection = CRS.from_user_input(dataset.crs)
            ruler_pixels = min(100, patch.width)
            map_distance = (extent[2] - extent[0]) * ruler_pixels / patch.width
            if projection.is_geographic:
                midlat = (extent[1] + extent[3]) / 2
                ruler_km = spherical_distance(midlat, extent[0], midlat, extent[0] + map_distance)
                ruler_basis = f"great-circle distance at latitude {midlat:.3g}"
            else:
                ruler_km = map_distance * projection.axis_info[0].unit_conversion_factor / 1000
                ruler_basis = "projected map distance"
    else:
        layers = [_collection(path) for path in paths]
        features = [feature for layer in layers for feature in layer["features"]]
        field = params.get("classification_field")
        classes, palette = (
            [],
            [
                "#dc2626",
                "#2563eb",
                "#059669",
                "#d97706",
                "#7c3aed",
                "#db2777",
                "#0891b2",
                "#4d7c0f",
                "#9333ea",
                "#475569",
                "#be123c",
                "#a16207",
            ],
        )
        if field:
            if any(
                (
                    field not in (feature.get("properties") or {})
                    or feature["properties"][field] is None
                    for feature in features
                )
            ):
                raise ValueError("Every rendered feature requires its classification attribute")
            classes = sorted(set((str(feature["properties"][field]) for feature in features)))
            if len(classes) > len(palette):
                raise UnsupportedOperation(
                    "Categorical rendering supports at most 12 explicit classes; pre-bin continuous fields"
                )
        if any(
            (
                (row.get("geometry") or {}).get("type")
                not in {"Point", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
                for row in features
            )
        ):
            raise UnsupportedOperation("Unsupported geometry type for static rendering")
        coordinates = [
            point for row in features for point in _coordinates(row["geometry"]["coordinates"])
        ]
        if not coordinates:
            raise ValueError("Cannot render an empty geometry collection")
        xs, ys = zip(*coordinates)
        extent, crs = ([min(xs), min(ys), max(xs), max(ys)], "EPSG:4326")
        if extent[2] - extent[0] > 180:
            raise UnsupportedOperation("Antimeridian-spanning maps need projection handling")
        width, height = (max(extent[2] - extent[0], 1e-06), max(extent[3] - extent[1], 1e-06))
        scale = min(840 / width, 500 / height)

        def pixel(point):
            return (80 + (point[0] - extent[0]) * scale, 600 - (point[1] - extent[1]) * scale)

        for feature in features:
            geometry = feature["geometry"]
            kind, coords = (geometry["type"], geometry["coordinates"])
            color = palette[classes.index(str(feature["properties"][field]))] if field else None
            if kind == "Point":
                x, y = pixel(coords)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color or "#b91c1c")
            elif kind in {"LineString", "MultiLineString"}:
                for line in [coords] if kind == "LineString" else coords:
                    draw.line([pixel(point) for point in line], fill=color or "#1d4ed8", width=3)
            else:
                for polygon in [coords] if kind == "Polygon" else coords:
                    for index, ring in enumerate(polygon):
                        draw.polygon(
                            [pixel(point) for point in ring],
                            fill=color or "#8bc4e8" if index == 0 else "white",
                            outline="#1e3a5f",
                        )
        feature_count = len(features)
        render_info = {"feature_count": feature_count}
        if field:
            render_info.update(classification_field=field, classes=classes)
            for index, label in enumerate(classes):
                draw.rectangle((780, 55 + index * 20, 792, 67 + index * 20), fill=palette[index])
                draw.text((798, 55 + index * 20), label, fill="black")
        ruler_pixels = min(100, max(1, width * scale))
        midlat = (extent[1] + extent[3]) / 2
        ruler_km = spherical_distance(
            midlat, extent[0], midlat, min(180, extent[0] + ruler_pixels / scale)
        )
        ruler_basis = f"great-circle distance at latitude {midlat:.3g}"
    draw.line((60, 615, 60 + ruler_pixels, 615), fill="black", width=2)
    for tick in (60, 60 + ruler_pixels):
        draw.line((tick, 610, tick, 620), fill="black", width=2)
    draw.text((65 + ruler_pixels, 607), f"{ruler_km:.3g} km", fill="black")
    draw.line((955, 95, 955, 50), fill="black", width=2)
    draw.polygon([(955, 42), (949, 56), (961, 56)], fill="black")
    draw.text((951, 25), "N", fill="black")
    render_info.update(
        scale_bar_km=ruler_km, scale_bar_basis=ruler_basis, north_arrow="map/grid north"
    )
    draw.text((60, 20), title, fill="black")
    draw.text((60, 635), f"CRS: {crs}; extent: {extent}", fill="black")
    draw.text((60, 660), str(render_info), fill="black")
    stem = re.sub("[^A-Za-z0-9_-]+", "_", title).strip("_") or "spatial_map"
    stem += "_" + uuid4().hex
    directory = output_root / "maps"
    directory.mkdir(parents=True, exist_ok=True)
    png, webpage = (directory / (stem + ".png"), directory / (stem + ".html"))
    image.save(png)
    webpage.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><h1>{html.escape(title)}</h1><img src='{png.name}' alt='Rendered spatial input'><p>{html.escape(str(render_info))}</p>",
        encoding="utf-8",
    )
    return {
        "map_png": str(png),
        "map_html": str(webpage),
        "title": title,
        "crs": crs,
        "extent": extent,
        "sources": list(map(str, paths)),
        "method": "static spatial data rendering",
        **render_info,
    }
