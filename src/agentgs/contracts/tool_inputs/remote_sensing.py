"""Remote sensing tool argument declarations; field policies are preserved."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import ArtifactReference, Number, Numbers, Paths, RasterPair, Text, ToolArguments


class RemoteSensingArguments(ToolArguments):
    @model_validator(mode="after")
    def check_series_lengths(self):
        data = self.model_dump(exclude_unset=True)
        if (
            self.tool_name == "RemoteSensingLinearTrend"
            and data.get("x") is not None
            and len(data["x"]) != len(data["y"])
        ):
            raise ValueError("x and y must have equal lengths")
        if self.tool_name == "RemoteSensingBatchNDSI" and len(data["green_file_list"]) != len(
            data["swir_file_list"]
        ):
            raise ValueError("green_file_list and swir_file_list must have equal lengths")
        return self


def register(add):
    add("RemoteSensingListFiles", {"dir_path": Text}, ("dir_path",))
    reference_files = Annotated[list[Text | ArtifactReference], Field(min_length=1)]
    batch_options = {
        "file_list": reference_files,
        "output_path": Text,
        "threshold": Number,
        "comparison": Literal["above", "below"],
        "relative_to_each_mean": bool,
        "relative_factor": Number,
        "empty_acquisition_policy": Literal["error", "include_zero"],
        "rs_date_window": Annotated[list[Text], Field(min_length=2, max_length=2)],
    }
    add(
        "RemoteSensingBatchImageMean",
        {
            **batch_options,
            "rs_statistic_mode": Literal["legacy", "per_scene_statistics", "pixelwise_composite"],
        },
        ("file_list",),
    )
    add("RemoteSensingBatchHotspotPercentage", batch_options, ("file_list", "threshold"))
    add("RemoteSensingBatchImageSum", {"file_list": reference_files}, ("file_list",))
    add(
        "RemoteSensingTiffAverage",
        {"file_list": reference_files, "output_path": Text},
        ("file_list",),
    )
    add(
        "RemoteSensingBatchNDSI",
        {"green_file_list": Paths, "swir_file_list": Paths, "output_path_list": Paths},
        ("green_file_list", "swir_file_list"),
    )
    pairs = {
        "NDWI": ("input_swir_path", "input_nir_path", "input_swir_paths", "input_nir_paths"),
        "NBR": ("input_nir_path", "input_swir_path", "input_nir_paths", "input_swir_paths"),
        "NDVI": ("nir_band_file", "red_band_file", "nir_band_files", "red_band_files"),
        "CloudMask": ("sr_band_path", "qa_pixel_path", "sr_band_paths", "qa_pixel_paths"),
        "TiffDifference": ("image_a_path", "image_b_path", "image_a_paths", "image_b_paths"),
    }
    for suffix, keys in pairs.items():
        types = {key: Text if i < 2 else Paths for i, key in enumerate(keys)}
        types["output_file" if suffix == "NDVI" else "output_path"] = Text
        add("RemoteSensing" + suffix, types, exclusive=(keys[:2], keys[2:]))
    add(
        "RemoteSensingNDTI",
        {"input_pairs": Annotated[list[RasterPair], Field(min_length=1)]},
        ("input_pairs",),
    )
    add(
        "RemoteSensingBatchHotspotRaster",
        {"file_list": Paths, "threshold": Number, "output_path_list": Paths, "comparison": Text},
        ("file_list", "threshold"),
    )
    add(
        "RemoteSensingCountAboveThreshold", {"file_path": Text, "threshold": Number}, ("file_path",)
    )
    add(
        "RemoteSensingExtremeSnowLossPercentage",
        {"binary_map_path": Text, "threshold": Number},
        ("binary_map_path",),
    )
    add(
        "RemoteSensingSubtract",
        {"img1_path": Text, "img2_path": Text, "output_path": Text, "nodata_as_zero": bool},
        ("img1_path", "img2_path"),
    )
    add(
        "RemoteSensingFireIncreaseMap",
        {"change_image_path": Text, "threshold": Number, "output_path": Text},
        ("change_image_path", "threshold"),
    )
    for suffix in (
        "Mean",
        "MaxValueAndIndex",
        "MinValueAndIndex",
        "SensSlope",
        "CoefficientOfVariation",
        "Kurtosis",
        "MannKendallTest",
        "Argmax",
    ):
        add("RemoteSensing" + suffix, {"x": Numbers}, ("x",))
    for suffix in ("PercentageChange", "Difference"):
        add("RemoteSensing" + suffix, {"a": Number, "b": Number}, ("a", "b"))
    add(
        "RemoteSensingDivision",
        {"a": Number, "b": Number, "a_values": Numbers, "b_values": Numbers},
        exclusive=(("a", "b"), ("a_values", "b_values")),
    )
    add("RemoteSensingKelvinToCelsius", {"kelvin": Number}, ("kelvin",))
    add("RemoteSensingLinearTrend", {"y": Numbers, "x": Numbers}, ("y",))
