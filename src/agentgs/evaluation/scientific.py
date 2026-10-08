"""Portable scientific fixtures and observations for source differential checks."""

import hashlib
import json
from pathlib import Path


def create_inputs(root):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    folder = root / "data/synthetic"
    folder.mkdir(parents=True)
    arrays = {
        "first.tif": [[2, 4, -9999], [6, 8, 10]],
        "second.tif": [[1, 2, 3], [4, -9999, 6]],
        "scaled.tif": [[0, 2, -9999], [4, 6, 8]],
        "shifted.tif": [[1, 2, 3], [4, 5, 6]],
    }
    for name, array in arrays.items():
        with rasterio.open(
            folder / name,
            "w",
            driver="GTiff",
            height=2,
            width=3,
            count=1,
            dtype="float64",
            crs="EPSG:4326",
            nodata=-9999,
            transform=from_origin(100 if name == "shifted.tif" else 0, 2, 1, 1),
        ) as target:
            target.write(np.asarray(array, dtype="float64"), 1)
            if name == "scaled.tif":
                target.scales, target.offsets = (0.5,), (1.0,)
                target.update_tags(1, valid_range="0 6")
    points = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coordinate},
                "properties": {"value": value, "age": value, "case_id": str(i)},
            }
            for i, (coordinate, value) in enumerate(
                [([0.5, 1.5], 1), ([1.5, 0.5], 3), ([2.5, 1.5], 5)]
            )
        ],
    }
    (folder / "points.geojson").write_text(json.dumps(points))
    (folder / "zones.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]]],
                        },
                        "properties": {"zone_id": "A"},
                    }
                ],
            }
        )
    )
    return folder


def bind_inputs(value, folder):
    if isinstance(value, dict):
        return {k: bind_inputs(v, folder) for k, v in value.items()}
    if isinstance(value, list):
        return [bind_inputs(v, folder) for v in value]
    if isinstance(value, str):
        return value.replace("$DATA", str(folder))
    return value


def observe_result(value, inputs, outputs):
    """Compare values and produced raster content, ignoring task-owned locations."""
    products = {}

    def walk(item):
        if isinstance(item, dict):
            return {k: walk(v) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(v) for v in item]
        if isinstance(item, str) and item.startswith(str(inputs)):
            return "$DATA/" + str(Path(item).relative_to(inputs))
        if isinstance(item, str) and item.startswith(str(outputs)):
            path = Path(item)
            if path not in products:
                products[path] = f"$OUTPUT/{len(products)}{path.suffix}"
            return products[path]
        return item

    result = walk(value)
    rasters, vectors, images = {}, {}, {}
    for path, name in list(products.items()):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            import numpy as np
            import rasterio

            with rasterio.open(path) as source:
                array = source.read(1, masked=True).astype("float64").filled(np.nan)
                rasters[name] = {
                    "values": [
                        [float(v) if np.isfinite(v) else None for v in row] for row in array
                    ],
                    "crs": str(source.crs),
                    "transform": list(source.transform),
                    "scales": list(source.scales),
                    "offsets": list(source.offsets),
                }
        elif path.is_file() and path.suffix.lower() in {".json", ".geojson"}:
            vectors[name] = walk(json.loads(path.read_text()))
        elif path.is_file() and path.suffix.lower() == ".png":
            from PIL import Image

            with Image.open(path) as source:
                # The final line embeds absolute input paths. Compare its
                # scientific fields in the structured result, and compare the
                # map, title, ruler, north arrow and CRS here.
                bounds = (0, 0, source.width, min(650, source.height))
                images[name] = {
                    "size": list(source.size),
                    "mode": source.mode,
                    "compared_pixel_bounds": list(bounds),
                    "pixels_sha256": hashlib.sha256(source.crop(bounds).tobytes()).hexdigest(),
                }
    return {"result": result, "rasters": rasters, "vectors": vectors, "images": images}
