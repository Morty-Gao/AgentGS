from dataclasses import dataclass
from typing import Callable

from pydantic import Field

from agentgs.contracts.models import Model, ToolSpec
from agentgs.runtime.tools.domains import geoscience as geo
from agentgs.runtime.tools.domains.earthquake import CatalogQuery, EventQuery


class CodeInput(Model):
    code: str = Field(min_length=1, max_length=100000)


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    input_type: type
    handler: Callable | None
    source_type: str = "raw_compute"


def builtin_tools():
    rows = [
        (
            "GeoHaversineDistance",
            "Great-circle distance between WGS84 points",
            geo.Points,
            geo.distance,
        ),
        ("GeoBearing", "Initial spherical bearing between WGS84 points", geo.Points, geo.bearing),
        (
            "GeoMagnitudeToEnergy",
            "Estimate earthquake energy from moment magnitude",
            geo.Magnitude,
            geo.energy,
        ),
        (
            "GeoWGS84BBoxAreaKm2",
            "Spherical area of an ordered WGS84 bounding box",
            geo.BBox,
            geo.area,
        ),
        ("GeoCheckCRSConsistency", "Compare normalized CRS labels", geo.CRSList, geo.crs),
        ("GeoDecimalToDMS", "Convert decimal degrees to signed DMS", geo.DecimalDegrees, geo.dms),
        (
            "DeepEarthQueryUSGSCatalog",
            "Filter and rank the frozen USGS M7+ catalog",
            CatalogQuery,
            None,
        ),
        ("DeepEarthGetUSGSEvent", "Find an event in the frozen USGS M7+ catalog", EventQuery, None),
    ]
    rows.append(
        (
            "ExecutePython",
            "Run Python in the configured code sandbox (network disabled with bwrap)",
            CodeInput,
            None,
        )
    )
    return {
        name: RegisteredTool(
            ToolSpec(
                id=name,
                description=description,
                input_schema=model.model_json_schema(),
                effects="write" if name == "ExecutePython" else "read",
            ),
            model,
            handler,
        )
        for name, description, model, handler in rows
    }
