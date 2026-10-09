"""Spatial tool argument declarations; field policies are preserved."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .base import (
    BBox,
    Filters,
    Identifier,
    Identifiers,
    Latitude,
    Longitude,
    Number,
    Paths,
    PositiveInt,
    Text,
    ToolArguments,
)


class SpatialArguments(ToolArguments):
    @model_validator(mode="after")
    def check_raster_operation(self):
        data = self.model_dump(exclude_unset=True)
        if (
            self.tool_name.endswith("RasterAlgebra")
            and data.get("operation") != "mask"
            and data.get("scalar") is None
        ):
            raise ValueError("Arithmetic raster operations require scalar")
        return self


def register(add):
    for prefix, id_key in (("DeepTime", "series_ids"), ("CCS", "case_ids")):
        single_id = "series_id" if prefix == "DeepTime" else "case_id"
        add(
            prefix + "DecimalToDMS",
            {
                "value": Number,
                "field": Literal["lat", "lon"],
                single_id: Identifier,
                "facility_id": Identifier,
                "input_assets": Paths,
            },
            ("field",),
            (("value",), (single_id,), ("facility_id",)),
        )
        for suffix in ("HaversineDistance", "Bearing"):
            add(
                prefix + suffix,
                {
                    "lat1": Latitude,
                    "lon1": Longitude,
                    "lat2": Latitude,
                    "lon2": Longitude,
                    id_key: Identifiers,
                    "input_assets": Paths,
                },
                alternatives=(("lat1", "lon1", "lat2", "lon2"), (id_key,)),
            )
        add(prefix + "BBoxArea", {"bbox": BBox}, ("bbox",))
        basic = {
            "paths": Paths,
            "input_assets": Paths,
            "lon_field": Text,
            "lat_field": Text,
            "source_crs": Text,
            "target_crs": Text,
        }
        for suffix in (
            "StandardizeSpatialData",
            "ValidateSpatialData",
            "NearestNeighbor",
            "Buffer",
            "Overlay",
            "BatchVector",
        ):
            types = dict(basic)
            if suffix == "ValidateSpatialData":
                types.update(standardized_layer=Text, standardized_points=Text)
            if suffix == "NearestNeighbor":
                types["neighbor_count"] = PositiveInt
            if suffix == "Buffer":
                types["distance_km"] = Annotated[Number, Field(gt=0)]
            add(
                prefix + suffix,
                types,
                alternatives=(
                    ("paths",),
                    ("input_assets",),
                    ("standardized_layer",),
                    ("standardized_points",),
                )
                if suffix == "ValidateSpatialData"
                else (("paths",), ("input_assets",)),
            )
        add(
            prefix + "InterpolateIDW",
            {
                **basic,
                "points_path": Text,
                "point_path": Text,
                "value_field": Text,
                "grid_size": PositiveInt,
                "bbox": BBox,
                "power": Annotated[Number, Field(gt=0)],
                "date_field": Text,
                "month": Text,
                "filters": Filters,
                "allow_temporal_mixing": bool,
            },
            ("value_field",),
            (("points_path",), ("point_path",), ("paths",)),
        )
        add(
            prefix + "RasterAlgebra",
            {
                "raster_path": Text,
                "paths": Paths,
                "operation": Literal["mask", "add", "subtract", "multiply", "divide"],
                "minimum": Number,
                "maximum": Number,
                "scalar": Number,
                "band": PositiveInt,
            },
            ("operation",),
            (("raster_path",), ("paths",)),
        )
        add(
            prefix + "ZonalStatistics",
            {
                "raster_path": Text,
                "zones_path": Text,
                "table_path": Text,
                "paths": Paths,
                "zone_field": Text,
                "value_field": Text,
                "band": PositiveInt,
                "operation": Literal["mean", "sum", "min", "max", "count"],
                "raster_crs": Text,
                "zones_crs": Text,
                "variable": Text,
                "units": Text,
                "month": Text,
            },
            alternatives=(
                ("raster_path", "zones_path"),
                ("table_path", "zone_field", "value_field"),
                ("paths",),
            ),
        )
        add(
            prefix + "RenderMap",
            {
                **basic,
                "raster_path": Text,
                "vector_path": Text,
                "title": Text,
                "classification_field": Text,
                "band": PositiveInt,
            },
            alternatives=(("raster_path",), ("vector_path",), ("paths",)),
        )
    add(
        "CCSCheckCRSConsistency",
        {"crs_values": Paths, "input_assets": Paths},
        alternatives=(("crs_values",), ("input_assets",)),
    )
