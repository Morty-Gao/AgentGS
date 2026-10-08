"""Read physical raster observations using metadata, never filenames or task IDs."""

from __future__ import annotations

import math
import re

from agentgs.runtime.tools.scientific.spatial import UnsupportedOperation

MOD09_RANGE_SOURCE = "https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/5/MOD09GA.fs"


def read_scientific_raster(path, *, band=None):
    import numpy as np
    import rasterio

    with rasterio.open(path) as source:
        if band is None and source.count != 1:
            raise UnsupportedOperation(
                "Scientific raster tools require one explicit band; extract the desired band before execution"
            )
        band = 1 if band is None else band
        if type(band) is not int or not 1 <= band <= source.count:
            raise ValueError("band must identify an existing raster band")
        if np.issubdtype(np.dtype(source.dtypes[band - 1]), np.complexfloating):
            raise UnsupportedOperation(
                "Scientific raster tools require real-valued observations, not complex samples"
            )
        band_tags = {key.casefold(): value for key, value in source.tags(band).items()}
        dataset_tags = {key.casefold(): value for key, value in source.tags().items()}
        array = source.read(band, masked=True).astype("float64").filled(np.nan)
        scale, offset = (float(source.scales[band - 1]), float(source.offsets[band - 1]))
        scale_origin = "GDAL band scale/offset"
        tagged_scale = band_tags.get("scale_factor", band_tags.get("scale"))
        tagged_offset = band_tags.get("add_offset", band_tags.get("offset"))
        if tagged_scale is not None:
            proposed = float(tagged_scale)
            if scale != 1 and (not math.isclose(scale, proposed, rel_tol=1e-12)):
                raise UnsupportedOperation("Conflicting raster scale metadata")
            scale, scale_origin = (proposed, "explicit band scale tag")
        if tagged_offset is not None:
            proposed = float(tagged_offset)
            if offset != 0 and (not math.isclose(offset, proposed, rel_tol=1e-12)):
                raise UnsupportedOperation("Conflicting raster offset metadata")
            offset = proposed
        if not math.isfinite(scale) or not math.isfinite(offset) or scale == 0:
            raise UnsupportedOperation("Raster scale and offset must be finite, with nonzero scale")
        valid_range, range_origin = (None, "nodata and finite observations only")
        if "valid_range" in band_tags:
            values = re.findall("[-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?", band_tags["valid_range"])
            if len(values) != 2:
                raise UnsupportedOperation(
                    "Raster valid_range must declare exactly two raw-value bounds"
                )
            valid_range, range_origin = (list(map(float, values)), "explicit band valid_range")
        elif "valid_min" in band_tags and "valid_max" in band_tags:
            valid_range = [float(band_tags["valid_min"]), float(band_tags["valid_max"])]
            range_origin = "explicit band valid_min/valid_max"
        elif re.fullmatch("sur_refl_b0[1-7]", band_tags.get("name", "")) and (
            "MOD09GA" in dataset_tags.get("license", "").upper()
            or dataset_tags.get("shortname", "").upper() == "MOD09GA"
        ):
            valid_range, range_origin = ([-100.0, 16000.0], MOD09_RANGE_SOURCE)
        if valid_range is not None:
            low, high = valid_range
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise UnsupportedOperation("Invalid raster valid-range metadata")
            array[(array < low) | (array > high)] = np.nan
        observed = np.isfinite(array)
        with np.errstate(over="ignore", invalid="ignore"):
            array = array * scale + offset
        if np.any(observed & ~np.isfinite(array)):
            raise UnsupportedOperation(
                "Raster scale/offset overflow; observed cells cannot be silently converted to NoData"
            )
        array[~np.isfinite(array)] = np.nan
        policy = {
            "source": str(path),
            "band": band,
            "nodata": source.nodatavals[band - 1],
            "scale": scale,
            "offset": offset,
            "scale_source": scale_origin,
            "valid_raw_range_inclusive": valid_range,
            "range_source": range_origin,
            "quality_mask_applied": False,
        }
        if isinstance(policy["nodata"], float) and (not math.isfinite(policy["nodata"])):
            policy["nodata"] = "nonfinite nodata"
    return (array, policy)
