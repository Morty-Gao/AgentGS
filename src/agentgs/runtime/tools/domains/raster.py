"""raster scientific operations; baseline numerical decisions preserved.

Source: GeoToolEval runtime, SHA256 bae6bb172e6574be2524820dfa2e0a431d263957efbb716716b1dd1275ab8a0f.
Only algorithm functions are retained; resource ownership is task-scoped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from agentgs.runtime.tools.domains.shared import _result
from agentgs.runtime.tools.resources import _output_path, _resolve_path


def _read_raster(path: Path):
    from agentgs.runtime.tools.scientific.raster import read_scientific_raster

    return read_scientific_raster(path)[0]


def _write_raster(array, output: Path, reference: Path | None = None) -> Path:
    import numpy as np

    from agentgs.runtime.tools.scientific.spatial import ScientificResultError

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing artifact: {output}")
    if np.isinf(np.asarray(array)).any():
        raise ScientificResultError(
            "Raster arithmetic overflow; infinite cells cannot establish a valid artifact"
        )
    if not np.isfinite(np.asarray(array)).any():
        raise ScientificResultError(
            "A raster artifact must contain at least one finite observation"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rasterio

        if reference:
            with rasterio.open(reference) as source:
                profile = source.profile.copy()
            profile.update(driver="GTiff", dtype="float64", count=1, nodata=np.nan)
            with rasterio.open(output, "w", **profile) as target:
                target.write(np.asarray(array, dtype="float64"), 1)
                target.scales, target.offsets = ((1.0,), (0.0,))
            return output
    except ImportError:
        pass
    from PIL import Image

    Image.fromarray(np.asarray(array, dtype="float32"), mode="F").save(output)
    return output


def _raster_comparison(params, default="above"):
    comparison = str(params.get("comparison", default)).casefold()
    if comparison not in {"above", "below"}:
        raise ValueError("comparison must be above or below (strict > or <)")
    return comparison


def _raster_threshold(value, name="threshold"):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_matching_grids(paths):
    import rasterio

    from agentgs.runtime.tools.scientific.spatial import UnsupportedOperation

    expected = None
    for path in paths:
        with rasterio.open(path) as dataset:
            if dataset.count != 1 or dataset.crs is None:
                raise UnsupportedOperation(
                    "Raster arithmetic requires a single band and explicit CRS"
                )
            if (
                not all((math.isfinite(value) for value in tuple(dataset.transform)))
                or dataset.transform.determinant == 0
            ):
                raise UnsupportedOperation(
                    "Raster arithmetic requires a finite, invertible affine grid"
                )
            observed = (dataset.shape, dataset.crs, dataset.transform)
            if expected is None:
                expected = observed
            elif observed != expected:
                raise UnsupportedOperation(
                    "Raster arithmetic requires matching dimensions, CRS and affine grid; reproject explicitly first"
                )


def _file_list(params: dict[str, Any], key: str = "file_list") -> list[Path]:
    values = params.get(key) or params.get("files")
    if not values and key == "file_list":
        values = params.get("input_raster") or params.get("input_path")
    if isinstance(values, (str, Path)):
        values = [values]
    if not values:
        raise ValueError(f"Required non-empty {key} array was not provided")
    paths: list[Path] = []
    for value in values:
        resolved = _resolve_path(value)
        if resolved.is_dir():
            paths.extend(
                (
                    path.resolve()
                    for path in sorted(resolved.iterdir())
                    if path.is_file() and path.suffix.casefold() in (".tif", ".tiff")
                )
            )
        else:
            paths.append(resolved)
    if not paths:
        raise ValueError(f"No raster files were resolved from {key}")
    return list(dict.fromkeys(paths))


def _rs_list(params: dict[str, Any]) -> str:
    path_value = params.get("dir_path") or params.get("path") or params.get("input_resource")
    directory = _resolve_path(path_value)
    if directory.is_file():
        directory = directory.parent
    files = sorted((str(path.resolve()) for path in directory.iterdir() if path.is_file()))
    return _result(directory=directory, file_count=len(files), files=files)


def _rs_batch_stat(params: dict[str, Any], statistic: str) -> str:
    import numpy as np

    from agentgs.runtime.tools.scientific.raster import read_scientific_raster
    from agentgs.runtime.tools.scientific.spatial import ScientificResultError

    values = []
    policies, skipped = ([], [])
    empty_policy = str(params.get("empty_acquisition_policy", "error"))
    if empty_policy not in {"error", "skip"}:
        raise ValueError(
            "empty_acquisition_policy must be error or skip; absent observations are never physical zero"
        )
    threshold = _raster_threshold(params.get("threshold", 0))
    comparison = _raster_comparison(params)
    relative_to_each_mean = params.get("relative_to_each_mean", False)
    if not isinstance(relative_to_each_mean, bool):
        raise ValueError("relative_to_each_mean must be a JSON boolean")
    relative_factor = _raster_threshold(params.get("relative_factor", 1.0), "relative_factor")
    applied_thresholds: list[float] = []
    paths = _file_list(params)
    operation = params.get("rs_statistic_mode", "legacy") if statistic == "mean" else "legacy"
    if operation not in {"legacy", "per_scene_statistics", "pixelwise_composite"}:
        raise ValueError(
            "rs_statistic_mode must be legacy, per_scene_statistics or pixelwise_composite"
        )
    produce_composite = statistic == "mean" and operation != "per_scene_statistics"
    if produce_composite:
        _require_matching_grids(paths)
    mean_total = mean_valid = None
    mean_source = None
    for path in paths:
        array, policy = read_scientific_raster(path)
        policies.append(policy)
        finite = array[np.isfinite(array)]
        if not finite.size:
            if empty_policy == "skip":
                skipped.append({"path": str(path), "reason": "no_finite_observations"})
                continue
            raise ScientificResultError("No finite observations in raster: " + str(path))
        if statistic == "mean":
            value = float(finite.mean())
            if produce_composite and mean_total is None:
                mean_total = np.zeros_like(array, dtype="float64")
                mean_valid = np.zeros_like(array, dtype="int32")
                mean_source = path
            if produce_composite and array.shape == mean_total.shape:
                mask = np.isfinite(array)
                mean_total[mask] += array[mask]
                mean_valid[mask] += 1
        elif statistic == "sum":
            value = float(finite.sum())
        else:
            local_threshold = (
                float(finite.mean()) * relative_factor
                if relative_to_each_mean and finite.size
                else threshold
            )
            local_threshold = _raster_threshold(local_threshold)
            applied_thresholds.append(local_threshold)
            mask = finite < local_threshold if comparison == "below" else finite > local_threshold
            value = float(mask.mean() * 100)
        values.append({"path": str(path), "value": value, "finite_pixels": int(finite.size)})
    result: dict[str, Any] = {
        "statistic": statistic,
        "threshold": threshold if statistic == "hotspot_percentage" else None,
        "thresholds": applied_thresholds if relative_to_each_mean else None,
        "comparison": comparison if statistic == "hotspot_percentage" else None,
        "values": values,
        "input_interpretation": policies,
        "empty_acquisition_policy": empty_policy,
        "skipped_acquisitions": skipped,
    }
    if not values:
        raise ScientificResultError(
            "No finite observations remain after the declared empty-acquisition policy"
        )
    if operation != "legacy":
        result["rs_statistic_mode"] = operation
    if (
        statistic == "mean"
        and mean_total is not None
        and (mean_valid is not None)
        and (mean_source is not None)
    ):
        average = np.divide(
            mean_total, mean_valid, out=np.full_like(mean_total, np.nan), where=mean_valid > 0
        )
        output = _output_path(params.get("output_path"), "batch_mean_average.tif")
        _write_raster(average, output, mean_source)
        result["average_raster"] = str(output)
    return _result(**result)


def _rs_tiff_average(params: dict[str, Any]) -> str:
    import numpy as np

    paths = _file_list(params)
    _require_matching_grids(paths)
    total = valid = None
    for path in paths:
        array = _read_raster(path)
        if total is None:
            total = np.zeros_like(array, dtype="float64")
            valid = np.zeros_like(array, dtype="int32")
        mask = np.isfinite(array)
        total[mask] += array[mask]
        valid[mask] += 1
    average = np.divide(total, valid, out=np.full_like(total, np.nan), where=valid > 0)
    output = _output_path(params.get("output_path") or params.get("output_file"), "average.tif")
    _write_raster(average, output, paths[0])
    return _result(
        output_path=output, input_count=len(paths), finite_pixels=int(np.isfinite(average).sum())
    )


def _normalized_difference(first: Path, second: Path, output: Path) -> str:
    import numpy as np

    from agentgs.runtime.tools.scientific.raster import read_scientific_raster

    _require_matching_grids([first, second])
    a, first_policy = read_scientific_raster(first)
    b, second_policy = read_scientific_raster(second)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = a + b
        numerator = a - b
    observed = np.isfinite(a) & np.isfinite(b)
    if np.any(observed & (~np.isfinite(denominator) | ~np.isfinite(numerator))):
        raise ValueError("Normalized-difference arithmetic overflow")
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(denominator) & (denominator != 0)
    result = np.divide(numerator, denominator, out=np.full_like(a, np.nan), where=valid)
    _write_raster(result, output, first)
    return _result(
        output_path=output,
        finite_pixels=int(np.isfinite(result).sum()),
        mean=float(np.nanmean(result)),
        input_interpretation=[first_policy, second_policy],
        formula="(first - second) / (first + second)",
    )


def _rs_batch_ndsi(params: dict[str, Any]) -> str:
    outputs = []
    greens = _file_list(params, "green_file_list")
    swirs = _file_list(params, "swir_file_list")
    names = params.get("output_path_list") or []
    for index, (green, swir) in enumerate(zip(greens, swirs)):
        output = _output_path(
            names[index] if index < len(names) else None, f"{green.stem}_ndsi.tif"
        )
        outputs.append(json.loads(_normalized_difference(green, swir, output)))
    return _result(outputs=outputs)


def _rs_ndvi(params: dict[str, Any]) -> str:
    nir_values = params.get("nir_band_files") or params.get("input_nir_paths")
    red_values = params.get("red_band_files") or params.get("input_red_paths")
    if nir_values or red_values:
        if not isinstance(nir_values, list) or not isinstance(red_values, list):
            raise ValueError("NDVI batch mode requires nir_band_files and red_band_files arrays")
        if len(nir_values) != len(red_values) or not nir_values:
            raise ValueError("NDVI batch arrays must be non-empty and have equal length")
        outputs = []
        for nir_value, red_value in zip(nir_values, red_values):
            nir = _resolve_path(nir_value)
            red = _resolve_path(red_value)
            output = _output_path(None, f"{nir.stem}_ndvi.tif")
            outputs.append(json.loads(_normalized_difference(nir, red, output)))
        return _result(outputs=outputs, input_count=len(outputs))
    nir = _resolve_path(params.get("nir_band_file") or params.get("input_nir_path"))
    red = _resolve_path(params.get("red_band_file") or params.get("input_red_path"))
    return _normalized_difference(
        nir, red, _output_path(params.get("output_file") or params.get("output_path"), "ndvi.tif")
    )


def _rs_nbr(params: dict[str, Any]) -> str:
    nir_values = params.get("input_nir_paths")
    swir_values = params.get("input_swir_paths")
    if nir_values or swir_values:
        if not isinstance(nir_values, list) or not isinstance(swir_values, list):
            raise ValueError("NBR batch mode requires input_nir_paths and input_swir_paths arrays")
        if len(nir_values) != len(swir_values) or not nir_values:
            raise ValueError("NBR batch arrays must be non-empty and have equal length")
        outputs = []
        for nir_value, swir_value in zip(nir_values, swir_values):
            nir = _resolve_path(nir_value)
            swir = _resolve_path(swir_value)
            output = _output_path(None, f"{nir.stem}_nbr.tif")
            outputs.append(json.loads(_normalized_difference(nir, swir, output)))
        return _result(outputs=outputs, input_count=len(outputs))
    nir = _resolve_path(params["input_nir_path"])
    swir = _resolve_path(params["input_swir_path"])
    return _normalized_difference(nir, swir, _output_path(params.get("output_path"), "nbr.tif"))


def _rs_ndwi(params: dict[str, Any]) -> str:
    green_values = params.get("input_swir_paths") or params.get("green_band_files")
    nir_values = params.get("input_nir_paths") or params.get("nir_band_files")
    if green_values or nir_values:
        if not isinstance(green_values, list) or not isinstance(nir_values, list):
            raise ValueError("NDWI batch mode requires input_swir_paths and input_nir_paths arrays")
        if len(green_values) != len(nir_values) or not green_values:
            raise ValueError("NDWI batch arrays must be non-empty and have equal length")
        outputs = []
        for green_value, nir_value in zip(green_values, nir_values):
            green = _resolve_path(green_value)
            nir = _resolve_path(nir_value)
            output = _output_path(None, f"{green.stem}_ndwi.tif")
            outputs.append(json.loads(_normalized_difference(green, nir, output)))
        return _result(outputs=outputs, input_count=len(outputs))
    green = _resolve_path(params.get("input_swir_path") or params.get("green_band_file"))
    nir = _resolve_path(params.get("input_nir_path") or params.get("nir_band_file"))
    return _normalized_difference(green, nir, _output_path(params.get("output_path"), "ndwi.tif"))


def _rs_ndti(params: dict[str, Any]) -> str:
    outputs = []
    for pair in params.get("input_pairs", []):
        green = _resolve_path(pair["input_green_path"])
        red = _resolve_path(pair["input_red_path"])
        output = _output_path(pair.get("output_filename"), f"{green.stem}_ndti.tif")
        outputs.append(json.loads(_normalized_difference(green, red, output)))
    return _result(outputs=outputs)


def _rs_cloud_mask(params: dict[str, Any]) -> str:
    import numpy as np

    source_values = params.get("sr_band_paths")
    qa_values = params.get("qa_pixel_paths")
    if source_values or qa_values:
        if not isinstance(source_values, list) or not isinstance(qa_values, list):
            raise ValueError(
                "Cloud-mask batch mode requires sr_band_paths and qa_pixel_paths arrays"
            )
        if len(source_values) != len(qa_values) or not source_values:
            raise ValueError("Cloud-mask batch arrays must be non-empty and have equal length")
        outputs = []
        for source_value, qa_value in zip(source_values, qa_values):
            source = _resolve_path(source_value)
            qa_path = _resolve_path(qa_value)
            band, qa = (_read_raster(source), _read_raster(qa_path).astype("uint32"))
            clear = (qa & 1 << 3 == 0) & (qa & 1 << 4 == 0)
            masked = np.where(clear, band, np.nan)
            output = _output_path(None, f"{source.stem}_cloud_masked.tif")
            _write_raster(masked, output, source)
            outputs.append(
                {
                    "output_path": str(output),
                    "source_path": str(source),
                    "qa_pixel_path": str(qa_path),
                    "clear_percentage": float(clear.mean() * 100),
                }
            )
        return _result(outputs=outputs, input_count=len(outputs))
    source = _resolve_path(params["sr_band_path"])
    qa_path = _resolve_path(params["qa_pixel_path"])
    band, qa = (_read_raster(source), _read_raster(qa_path).astype("uint32"))
    clear = (qa & 1 << 3 == 0) & (qa & 1 << 4 == 0)
    masked = np.where(clear, band, np.nan)
    output = _output_path(params.get("output_path"), "cloud_masked.tif")
    _write_raster(masked, output, source)
    return _result(output_path=output, clear_percentage=float(clear.mean() * 100))


def _rs_hotspot_raster(params: dict[str, Any]) -> str:
    import numpy as np

    paths = _file_list(params)
    threshold = float(params["threshold"])
    names = params.get("output_path_list") or []
    outputs = []
    for index, path in enumerate(paths):
        array = _read_raster(path)
        comparison = str(params.get("comparison", "below")).lower()
        binary = np.where(
            np.isfinite(array),
            (array < threshold if comparison == "below" else array > threshold).astype("float32"),
            np.nan,
        )
        output = _output_path(
            names[index] if index < len(names) else None, f"hotspot_{index:03d}.tif"
        )
        _write_raster(binary, output, path)
        outputs.append(str(output))
    return _result(outputs=outputs, threshold=threshold)


def _rs_difference(params: dict[str, Any], subtract_keys: bool = False) -> str:
    import numpy as np

    def subtract(first_path, second_path):
        first_array = _read_raster(first_path)
        second_array = _read_raster(second_path)
        if bool(params.get("nodata_as_zero", False)):
            first_array = np.nan_to_num(first_array, nan=0.0)
            second_array = np.nan_to_num(second_array, nan=0.0)
        return first_array - second_array

    first_key, second_key = (
        ("img1_path", "img2_path") if subtract_keys else ("image_a_path", "image_b_path")
    )
    if not subtract_keys and (params.get("image_a_paths") or params.get("image_b_paths")):
        first_values = params.get("image_a_paths")
        second_values = params.get("image_b_paths")
        if not isinstance(first_values, list) or not isinstance(second_values, list):
            raise ValueError(
                "TiffDifference batch mode requires image_a_paths and image_b_paths arrays"
            )
        if not first_values or len(first_values) != len(second_values):
            raise ValueError("TiffDifference batch arrays must be non-empty and have equal length")
        outputs = []
        for first_value, second_value in zip(first_values, second_values):
            first, second = (_resolve_path(first_value), _resolve_path(second_value))
            array = subtract(first, second)
            output = _output_path(None, f"{first.stem}_to_{second.stem}_difference.tif")
            _write_raster(array, output, first)
            outputs.append(
                {
                    "output_path": str(output),
                    "image_a_path": str(first),
                    "image_b_path": str(second),
                    "mean": float(np.nanmean(array)),
                }
            )
        return _result(outputs=outputs, input_count=len(outputs))
    first, second = (_resolve_path(params[first_key]), _resolve_path(params[second_key]))
    array = subtract(first, second)
    output = _output_path(params.get("output_path"), "difference.tif")
    _write_raster(array, output, first)
    return _result(output_path=output, mean=float(np.nanmean(array)))


def _rs_fire_map(params: dict[str, Any]) -> str:
    import numpy as np

    source = _resolve_path(params["change_image_path"])
    array = _read_raster(source)
    result = np.where(
        np.isfinite(array), (array > float(params["threshold"])).astype("float32"), np.nan
    )
    output = _output_path(params.get("output_path"), "fire_increase_map.tif")
    _write_raster(result, output, source)
    return _result(output_path=output, positive_pixels=int(np.nansum(result)))


def _rs_count_threshold(params: dict[str, Any]) -> str:
    import numpy as np

    path = _resolve_path(params.get("file_path") or params.get("binary_map_path"))
    values = _read_raster(path)
    finite = values[np.isfinite(values)]
    threshold = float(params.get("threshold", 0))
    return _result(
        count=int((finite >= threshold).sum()),
        finite_pixels=int(finite.size),
        percentage=float((finite >= threshold).mean() * 100),
    )


def _series(params: dict[str, Any]) -> list[float]:
    values = params.get("x") or params.get("values")
    return [float(value) for value in values]


def _rs_scalar(name: str, params: dict[str, Any]) -> str:
    import numpy as np

    if name == "RemoteSensingMean":
        return _result(value=float(np.mean(_series(params))))
    if name == "RemoteSensingDifference":
        return _result(value=float(params["a"]) - float(params["b"]))
    if name == "RemoteSensingDivision":
        if params.get("a_values") is not None or params.get("b_values") is not None:
            a_values = np.asarray(params.get("a_values"), dtype="float64")
            b_values = np.asarray(params.get("b_values"), dtype="float64")
            if a_values.ndim != 1 or b_values.ndim != 1 or a_values.size == 0:
                raise ValueError("Division array mode requires non-empty one-dimensional arrays")
            if a_values.shape != b_values.shape:
                raise ValueError("Division arrays must have equal length")
            ratios = np.divide(
                a_values, b_values, out=np.full_like(a_values, np.nan), where=b_values != 0
            )
            return _result(values=ratios.tolist(), count=int(ratios.size))
        return _result(value=float(params["a"]) / float(params["b"]))
    if name == "RemoteSensingPercentageChange":
        a, b = (float(params["a"]), float(params["b"]))
        return _result(value=(b - a) / a * 100)
    if name == "RemoteSensingKelvinToCelsius":
        return _result(value=float(params.get("kelvin", params.get("value"))) - 273.15)
    if name == "RemoteSensingLinearTrend":
        y_values = params.get("y") or params.get("values") or params.get("x")
        values = np.asarray([float(value) for value in y_values], dtype=float)
        explicit_x = params.get("x") if params.get("y") is not None else None
        if explicit_x is None:
            x = np.arange(values.size, dtype=float)
        else:
            x = np.asarray([float(value) for value in explicit_x], dtype=float)
            if x.size and np.nanmin(np.abs(x)) >= 1000:
                x = x - x[0]
        if x.size != values.size:
            raise ValueError("RemoteSensingLinearTrend requires x and y arrays of equal length")
        slope, intercept = np.polyfit(x, values, 1)
        return _result(slope=float(slope), intercept=float(intercept))
    values = np.asarray(_series(params), dtype=float)
    if name == "RemoteSensingMaxValueAndIndex":
        index = int(np.nanargmax(values))
        return _result(value=float(values[index]), index=index)
    if name == "RemoteSensingMinValueAndIndex":
        index = int(np.nanargmin(values))
        return _result(value=float(values[index]), index=index)
    if name == "RemoteSensingArgmax":
        return _result(index=int(np.nanargmax(values)))
    if name == "RemoteSensingSensSlope":
        slopes = [
            (values[j] - values[i]) / (j - i)
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        return _result(slope=float(np.median(slopes)))
    if name == "RemoteSensingCoefficientOfVariation":
        return _result(value=float(np.std(values) / np.mean(values)))
    if name == "RemoteSensingKurtosis":
        centered = values - values.mean()
        variance = np.mean(centered**2)
        pearson = float(np.mean(centered**4) / variance**2)
        return _result(kurtosis=pearson, excess_kurtosis=pearson - 3.0)
    if name == "RemoteSensingMannKendallTest":
        s = sum(
            (
                int(np.sign(values[j] - values[i]))
                for i in range(len(values))
                for j in range(i + 1, len(values))
            )
        )
        variance = len(values) * (len(values) - 1) * (2 * len(values) + 5) / 18
        z = (
            (s - 1) / math.sqrt(variance)
            if s > 0
            else (s + 1) / math.sqrt(variance)
            if s < 0
            else 0.0
        )
        return _result(
            s=s,
            z=z,
            p_value=math.erfc(abs(z) / math.sqrt(2)),
            trend="increasing" if s > 0 else "decreasing" if s < 0 else "no trend",
        )
    raise KeyError(name)
