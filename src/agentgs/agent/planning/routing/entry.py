from __future__ import annotations

import re

from agentgs.agent.planning.routing.common import (
    _canonicalize_remote_chain,
    _resource_parts,
    _unique,
)
from agentgs.agent.planning.routing.remote import _remote_plan
from agentgs.agent.planning.routing.spatial import _deep_time_or_ccs


def infer_tools(question: str) -> list[str]:
    clean, resource, locator, sources = _resource_parts(question)
    text = f"{clean} {resource}".casefold()
    if isinstance(resource, str) and (
        re.search("\\.(?:tif|tiff|npy)(?:$|[/?])", resource, re.IGNORECASE)
        or any(
            (
                token in text
                for token in (
                    "raster",
                    "geotiff",
                    "ndvi",
                    "ndsi",
                    "ndwi",
                    "nbr",
                    "ndti",
                    "sur_refl",
                    "qa_pixel",
                    "rainfall",
                    "nighttime light",
                    "vegetation",
                    "wildfire",
                    "fire intensity",
                    "maxfrp",
                    "burned area",
                    "snow cover",
                    "thermal",
                    "building volume",
                    "built_volume",
                    "land cover",
                )
            )
        )
    ):
        return _remote_plan(text, resource)
    semantic_terms = (
        "beach",
        "canal",
        "coast",
        "county",
        "district",
        "forest",
        "island",
        "lake",
        "lagoon",
        "locality",
        "municipality",
        "park",
        "reserve",
        "river",
        "spatial",
        "stream",
        "townland",
        "village",
        "water",
    )
    semantic_resource = (
        resource
        if isinstance(resource, str)
        else locator.get("resource")
        if isinstance(locator, dict)
        else None
    )
    if (
        isinstance(semantic_resource, str)
        and "/" not in semantic_resource
        and re.fullmatch("[a-z][a-z0-9]*(?:_[a-z0-9]+)+", semantic_resource)
        and any((term in text for term in semantic_terms))
    ):
        return ["GeoBigDataExecuteGeoSPARQL"]
    input_assets = " ".join(
        (
            str(value)
            for value in (resource.get("input_assets", []) if isinstance(resource, dict) else [])
        )
    ).casefold()
    if "biodeeptime" in input_assets:
        return _deep_time_or_ccs("DeepTime", clean.casefold(), locator)
    if "ccs_" in input_assets or "ccs-" in text:
        return _deep_time_or_ccs("CCS", clean.casefold(), locator)
    if any((token in sources for token in ("litho1.0", "semucb", "usgs-fdsn"))):
        if "litho_node_index" in locator or "semucb_depth_km" in locator:
            return ["DeepEarthGetUSGSEvent", "DeepEarthSampleEventModels"]
        if "litho1.0" in sources:
            return ["DeepEarthQueryLITHO1"]
        if "semucb" in sources:
            return ["DeepEarthQuerySEMUCB"]
        return ["DeepEarthQueryUSGSCatalog"]
    if any(
        (token in sources for token in ("opsd", "mrds csv", "gem", "world bank", "un population"))
    ):
        keyset = set(locator)
        mapping = (
            ({"solar_municipality", "wind_municipality"}, "EnergyFacilityDistance"),
            ({"dep_id_1", "dep_id_2"}, "EnergyMineralSiteDistance"),
            ({"bbox", "plant_name"}, "EnergyBBoxArea"),
            ({"capital", "energy_source"}, "EnergyNearestFacility"),
            ({"series", "year"}, "EnergyCapacityValue"),
            ({"mrds_dep_id", "plant_name"}, "EnergyNearestMineralSite"),
            ({"renewable_types", "status"}, "EnergyCountryRenewableSummary"),
            ({"series", "start_year", "end_year"}, "EnergyCapacityGrowth"),
            ({"population_year", "consumption_year"}, "EnergyTransitionMetrics"),
            ({"bbox", "region"}, "EnergyRegionalOverlaySummary"),
        )
        for required, tool in mapping:
            if required <= keyset:
                return [tool]
    if any((token in sources for token in ("socat", "glodap", "snapo"))):
        keyset = set(locator)
        if {"date", "latitude", "longitude"} <= keyset:
            return ["OceanMonthlyGridValue"]
        if {"depth_m", "expocode", "station"} <= keyset:
            return ["OceanProfileValue"]
        if {"cruise", "required_flags"} <= keyset:
            return ["OceanCruiseQC"]
        if {"glodap_expocode", "glodap_station", "snapo_cruise"} <= keyset:
            return ["OceanStationDistance"]
        if {"fCO2rec_flag", "year"} <= keyset:
            return ["OceanAnnualCarbonSummary"]
        if {"glodap_expocode", "glodap_station", "socat_date"} <= keyset:
            return ["OceanSurfaceDeepMatch"]
        if {"cruise", "deduplication_key"} <= keyset:
            return ["OceanProfileDeduplication"]
        if {"expocode", "station"} <= keyset:
            return ["OceanVerticalProfileDifference"]
        if {"bbox", "region"} <= keyset:
            return ["OceanRegionalMapSummary" if "map" in text else "OceanBBoxArea"]
    if "自然灾害" in text or ("filters" in locator and "operation" in locator):
        operation = str(locator.get("operation", "")).casefold()
        chain = ["NaturalHazardLoadEventTable", "NaturalHazardFilterRecords"]
        if operation == "count_records":
            chain.append("NaturalHazardCountRecords")
        elif operation in ("top_count", "top_sum"):
            chain.append("NaturalHazardGroupAggregate")
        elif operation == "sum":
            chain.append("NaturalHazardSumField")
        elif operation == "max_event":
            chain.append("NaturalHazardSelectMaxRecord")
        elif operation == "latest_date":
            chain.extend(("NaturalHazardBuildStartDate", "NaturalHazardSelectLatest"))
        return chain
    keyset = set(locator)
    climate: list[str] = []
    mineral: list[str] = []
    if "deep_age_kyr_bp" in keyset:
        climate = [
            "ClimateQueryDeepTime",
            "ClimateQueryPresentMean",
            "ClimateQueryFutureSSP",
            "ClimateCompareEpochs",
        ]
    elif "ages_kyr_bp" in keyset:
        climate = ["ClimateQueryDeepTime", "ClimateQueryDeepTime", "ClimateQueryFutureSSP"]
    elif "find" in keyset and "ssp" in keyset:
        climate = ["ClimateQueryDeepTime", "ClimateQueryFutureSSP", "ClimateCompareEpochs"]
    elif "year_start" in keyset and ("scenario" in keyset or "ssp" in keyset):
        climate = ["ClimateQueryPresentMean", "ClimateQueryFutureSSP", "ClimateCompareEpochs"]
    elif {"dataset", "age_kyr_bp"} <= keyset or "find" in keyset:
        climate = ["ClimateQueryDeepTime"]
    elif {"year_start", "year_end"} <= keyset:
        climate = ["ClimateQueryPresentMean"]
    elif {"scenario", "horizon_year"} <= keyset:
        climate = ["ClimateQueryFutureSSP"]
    if "dep_id" in keyset or re.search("\\bdep_id\\s*=", text):
        mineral = (
            ["MineralResourcesSearchDeposits", "MineralResourcesGetDeposit"]
            if "search" in text and {"country", "commodity"} <= keyset
            else ["MineralResourcesGetDeposit"]
        )
    elif {"country", "commodity"} <= keyset and "scenario" not in keyset:
        mineral = ["MineralResourcesSearchDeposits"]
    elif "top deposit" in text or "which countries have the most deposit" in text:
        mineral = ["MineralResourcesRankCommodityCountries"]
    elif "count metal-mineral records" in text or "count all metal records" in text:
        mineral = ["MineralResourcesCountrySummary"]
    if {"commodity", "horizon_year", "scenario"} <= keyset:
        return ["MineralResourcesRankCommodityCountries", "ClimateQueryFutureSSP"]
    if {"country", "commodity", "scenario"} <= keyset:
        return [
            "MineralResourcesCountrySummary",
            "MineralResourcesSearchDeposits",
            "ClimateQueryFutureSSP",
        ]
    if climate and (not mineral):
        return climate
    if mineral or climate:
        return _unique(mineral + climate)
    return []


def normalize_plan(
    proposed: list[str], inferred: list[str], available: list[str]
) -> tuple[list[str], str]:
    available_set = set(available)
    proposed = [tool for tool in proposed if tool in available_set]
    inferred = [tool for tool in inferred if tool in available_set]
    if proposed:
        if inferred and inferred[0] == "RemoteSensingListFiles":
            index_tools = {
                "RemoteSensingBatchNDSI",
                "RemoteSensingNDWI",
                "RemoteSensingNBR",
                "RemoteSensingNDVI",
                "RemoteSensingNDTI",
            }
            reduction_tools = {
                "RemoteSensingBatchImageMean",
                "RemoteSensingTiffAverage",
                "RemoteSensingMean",
            }
            proposed_positions = {tool: proposed.index(tool) for tool in set(proposed)}
            invalid_dependency_order = any(
                (
                    generator in proposed_positions
                    and any(
                        (
                            reducer in proposed_positions
                            and proposed_positions[reducer] < proposed_positions[generator]
                            for reducer in reduction_tools
                        )
                    )
                    for generator in index_tools
                )
            )
            if "RemoteSensingCloudMask" in proposed_positions and any(
                (
                    generator in proposed_positions
                    and proposed_positions[generator] < proposed_positions["RemoteSensingCloudMask"]
                    for generator in index_tools
                )
            ):
                invalid_dependency_order = True
            critical_tools = {
                "RemoteSensingTiffAverage",
                "RemoteSensingBatchHotspotPercentage",
                "RemoteSensingBatchHotspotRaster",
                "RemoteSensingExtremeSnowLossPercentage",
                "RemoteSensingLinearTrend",
                "RemoteSensingSensSlope",
                "RemoteSensingMannKendallTest",
                "RemoteSensingMaxValueAndIndex",
                "RemoteSensingMinValueAndIndex",
                "RemoteSensingArgmax",
                "RemoteSensingSubtract",
                "RemoteSensingFireIncreaseMap",
                "RemoteSensingCountAboveThreshold",
                "RemoteSensingDivision",
            }
            missing_critical_tool = any(
                (tool in inferred and tool not in proposed for tool in critical_tools)
            )
            specific_trend_conflict = "RemoteSensingLinearTrend" in proposed and any(
                (
                    tool in inferred
                    for tool in ("RemoteSensingSensSlope", "RemoteSensingMannKendallTest")
                )
            )
            cardinality_loss = any(
                (
                    inferred.count(tool) > proposed.count(tool)
                    for tool in critical_tools
                    if tool in inferred
                )
            )
            if (
                invalid_dependency_order
                or missing_critical_tool
                or specific_trend_conflict
                or cardinality_loss
            ):
                return (_canonicalize_remote_chain(inferred), "resource_validated_remote_plan")
            augmented = list(proposed)
            file_batch_tools = {
                "RemoteSensingBatchImageMean",
                "RemoteSensingBatchImageSum",
                "RemoteSensingBatchNDSI",
                "RemoteSensingBatchHotspotPercentage",
                "RemoteSensingBatchHotspotRaster",
            }
            if (
                any((tool in file_batch_tools for tool in augmented))
                and "RemoteSensingListFiles" not in augmented
            ):
                augmented.insert(0, "RemoteSensingListFiles")
            if "RemoteSensingMean" in inferred:
                producer_count = augmented.count("RemoteSensingBatchImageMean")
                target_count = producer_count if producer_count > 1 else 1
                missing = max(0, target_count - augmented.count("RemoteSensingMean"))
                if missing:
                    terminal_tools = {
                        "RemoteSensingDifference",
                        "RemoteSensingPercentageChange",
                        "RemoteSensingDivision",
                        "RemoteSensingMaxValueAndIndex",
                        "RemoteSensingMinValueAndIndex",
                    }
                    insert_at = next(
                        (index for index, tool in enumerate(augmented) if tool in terminal_tools),
                        len(augmented),
                    )
                    augmented[insert_at:insert_at] = ["RemoteSensingMean"] * missing
            return (_canonicalize_remote_chain(augmented), "resource_augmented_model_plan")
        single_call_query_tools = {
            "ClimateQueryDeepTime",
            "ClimateQueryPresentMean",
            "ClimateQueryFutureSSP",
        }
        if (
            len(proposed) > 1
            and len(set(proposed)) == 1
            and (proposed[0] in single_call_query_tools)
            and (inferred == [proposed[0]])
        ):
            return ([proposed[0]], "schema_cardinality_normalized_model_plan")
        return (_canonicalize_remote_chain(proposed), "model_plan")
    if not inferred:
        return (available[:1], "model_plan_fallback")
    if inferred[0] != "RemoteSensingListFiles":
        return (inferred, "resource_contract_fallback")
    plan = list(proposed)
    if not plan or plan[0] != "RemoteSensingListFiles":
        plan.insert(0, "RemoteSensingListFiles")
    for tool in inferred[1:]:
        if tool not in plan:
            plan.append(tool)
    return (_canonicalize_remote_chain(plan), "resource_augmented_model_plan")
