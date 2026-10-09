"""Lookup tool argument declarations; field policies are preserved."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from .base import BBox, Identifier, Latitude, Longitude, Number, Numbers, Paths, PositiveInt, Text


def register(add):
    add(
        "DeepEarthQueryLITHO1",
        {"location_key": Text, "node_index": int, "layer": Literal["moho", "lid"]},
        ("location_key",),
    )
    add(
        "DeepEarthQuerySEMUCB",
        {"location_key": Text, "depth_km": Number, "parameter": Text},
        ("location_key", "depth_km"),
    )
    add(
        "DeepEarthQueryUSGSCatalog",
        {
            "operation": Literal[
                "count",
                "largest",
                "max_magnitude",
                "rank_magnitude",
                "deepest",
                "max_depth",
                "total_energy",
                "sum_energy",
                "event",
                "lookup",
            ],
            "minimum_magnitude": Number,
            "year": int,
            "start": Text,
            "end_exclusive": Text,
            "event_id": Text,
            "event_count": int,
            "formula": Text,
            "start_year": int,
            "end_year": int,
            "min_magnitude": Number,
            "limit": PositiveInt,
        },
    )
    add("DeepEarthGetUSGSEvent", {"event_id": Text}, ("event_id",))
    add(
        "DeepEarthSampleEventModels",
        {"event_id": Text, "litho_node_index": int, "semucb_depth_km": Number},
        ("event_id",),
    )
    add(
        "MineralResourcesSearchDeposits", {"country": Text, "commodity": Text, "limit": PositiveInt}
    )
    add("MineralResourcesGetDeposit", {"dep_id": int}, ("dep_id",))
    add("MineralResourcesCountrySummary", {"country": Text, "limit": PositiveInt}, ("country",))
    add(
        "MineralResourcesRankCommodityCountries",
        {"commodity": Text, "limit": PositiveInt},
        ("commodity",),
    )
    add(
        "ClimateQueryDeepTime",
        {
            "dataset": Text,
            "age_kyr_bp": Number,
            "ages_kyr_bp": Numbers,
            "find": Literal["min_temp_friedrich", "min_temp", "coldest"],
        },
        alternatives=(("age_kyr_bp",), ("ages_kyr_bp",), ("find",)),
    )
    add("ClimateQueryPresentMean", {"year_start": int, "year_end": int}, ("year_start", "year_end"))
    add("ClimateQueryFutureSSP", {"scenario": Text, "horizon_year": int}, ("scenario",))
    add(
        "ClimateCompareEpochs",
        {"values": Annotated[list[JsonValue], Field(min_length=2)], "operation": Text},
        ("values",),
    )

    energy = {
        "FacilityDistance": (
            {"country": Text, "solar_municipality": str | None, "wind_municipality": str | None},
            ("country", "solar_municipality", "wind_municipality"),
        ),
        "MineralSiteDistance": ({"dep_id_1": int, "dep_id_2": int}, ("dep_id_1", "dep_id_2")),
        "BBoxArea": ({"bbox": BBox, "plant_name": Text}, ("bbox",)),
        "NearestFacility": (
            {"country": Text, "energy_source": Text, "capital": Text},
            ("country", "energy_source"),
        ),
        "CapacityValue": ({"series": Text, "year": int}, ("series", "year")),
        "NearestMineralSite": ({"plant_name": Text, "mrds_dep_id": int}, ("plant_name",)),
        "CountryRenewableSummary": (
            {"country": Text, "renewable_types": Paths, "status": Text},
            ("country",),
        ),
        "CapacityGrowth": (
            {"series": Text, "start_year": int, "end_year": int},
            ("series", "start_year", "end_year"),
        ),
        "TransitionMetrics": (
            {"country": Text, "population_year": int, "consumption_year": int},
            ("country",),
        ),
        "RegionalOverlaySummary": ({"region": Text, "bbox": BBox}, ("region", "bbox")),
    }
    ocean = {
        "MonthlyGridValue": (
            {"date": Text, "latitude": Latitude, "longitude": Longitude},
            ("date", "latitude", "longitude"),
        ),
        "ProfileValue": (
            {"expocode": Text, "station": Identifier, "depth_m": Number},
            ("expocode", "station"),
        ),
        "CruiseQC": ({"cruise": Text, "required_flags": list[Identifier]}, ("cruise",)),
        "StationDistance": (
            {"glodap_expocode": Text, "glodap_station": Identifier, "snapo_cruise": Text},
            ("glodap_expocode", "glodap_station", "snapo_cruise"),
        ),
        "BBoxArea": ({"bbox": BBox, "region": Text}, ("bbox",)),
        "AnnualCarbonSummary": ({"year": int, "fCO2rec_flag": int}, ("year",)),
        "VerticalProfileDifference": (
            {"expocode": Text, "station": Identifier},
            ("expocode", "station"),
        ),
        "RegionalMapSummary": ({"region": Text, "bbox": BBox, "year": int}, ("region", "bbox")),
        "SurfaceDeepMatch": (
            {"glodap_expocode": Text, "glodap_station": Identifier, "socat_date": Text},
            ("glodap_expocode", "glodap_station"),
        ),
        "ProfileDeduplication": ({"cruise": Text, "deduplication_key": Text | Paths}, ("cruise",)),
    }
    for prefix, definitions in (("Energy", energy), ("Ocean", ocean)):
        for suffix, (types, required) in definitions.items():
            add(prefix + suffix, {**types, "question_id": Identifier}, required)
