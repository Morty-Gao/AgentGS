"""Single domain dispatch authority derived from the frozen 101-tool catalog."""

from __future__ import annotations

import json
from typing import Any, Callable

from agentgs.contracts.arguments import parse_arguments
from agentgs.runtime.tools.domains.climate import (
    _climate_compare,
    _climate_deep,
    _climate_present,
    _climate_ssp,
)
from agentgs.runtime.tools.domains.deep_earth import (
    _deep_event_models,
    _deep_litho,
    _deep_semucb,
    _deep_usgs_event,
    _deep_usgs_query,
)
from agentgs.runtime.tools.domains.energy import (
    _energy_bbox,
    _energy_capacity,
    _energy_cross_nearest,
    _energy_facility_distance,
    _energy_growth,
    _energy_mineral_distance,
    _energy_mix,
    _energy_nearest,
    _energy_overlay,
    _energy_transition,
)
from agentgs.runtime.tools.domains.hazard import (
    _hazard_count,
    _hazard_dates,
    _hazard_filter,
    _hazard_group,
    _hazard_latest,
    _hazard_load,
    _hazard_max,
    _hazard_sum,
)
from agentgs.runtime.tools.domains.mineral import (
    _mineral_country_summary,
    _mineral_get,
    _mineral_rank_commodity,
    _mineral_search,
)
from agentgs.runtime.tools.domains.ocean import (
    _ocean_annual,
    _ocean_bbox,
    _ocean_dedup,
    _ocean_grid,
    _ocean_map,
    _ocean_match,
    _ocean_profile,
    _ocean_qc,
    _ocean_station_distance,
    _ocean_vertical,
)
from agentgs.runtime.tools.domains.raster import (
    _require_matching_grids,
    _rs_batch_ndsi,
    _rs_batch_stat,
    _rs_cloud_mask,
    _rs_count_threshold,
    _rs_difference,
    _rs_fire_map,
    _rs_hotspot_raster,
    _rs_list,
    _rs_nbr,
    _rs_ndti,
    _rs_ndvi,
    _rs_ndwi,
    _rs_scalar,
    _rs_tiff_average,
)
from agentgs.runtime.tools.domains.shared import _clean, _result
from agentgs.runtime.tools.domains.spatial import (
    _batch_vector,
    _bbox_tool,
    _bearing,
    _buffer_points,
    _crs_check,
    _decimal_to_dms,
    _distance,
    _interpolate_idw,
    _nearest_neighbor,
    _overlay,
    _raster_algebra,
    _render_map,
    _standardize_spatial,
    _validate_spatial,
    _zonal_stats,
)
from agentgs.runtime.tools.resources import _resolve_path


def _geosparql(params):
    raise RuntimeError("GeoQuestions uses the separately authorized typed snapshot adapter")


_GIS_SUFFIXES: dict[str, Callable[[dict[str, Any]], str]] = {
    "StandardizeSpatialData": _standardize_spatial,
    "ValidateSpatialData": _validate_spatial,
    "NearestNeighbor": _nearest_neighbor,
    "Buffer": _buffer_points,
    "Overlay": _overlay,
    "BatchVector": _batch_vector,
    "InterpolateIDW": _interpolate_idw,
    "RasterAlgebra": _raster_algebra,
    "ZonalStatistics": _zonal_stats,
    "RenderMap": _render_map,
}
_IMPLEMENTATIONS: dict[str, Callable[[dict[str, Any]], str]] = {
    "DeepEarthQueryLITHO1": _deep_litho,
    "DeepEarthQuerySEMUCB": _deep_semucb,
    "DeepEarthQueryUSGSCatalog": _deep_usgs_query,
    "DeepEarthGetUSGSEvent": _deep_usgs_event,
    "DeepEarthSampleEventModels": _deep_event_models,
    "GeoBigDataExecuteGeoSPARQL": _geosparql,
    "MineralResourcesSearchDeposits": _mineral_search,
    "MineralResourcesGetDeposit": _mineral_get,
    "MineralResourcesCountrySummary": _mineral_country_summary,
    "MineralResourcesRankCommodityCountries": _mineral_rank_commodity,
    "ClimateQueryDeepTime": _climate_deep,
    "ClimateQueryPresentMean": _climate_present,
    "ClimateQueryFutureSSP": _climate_ssp,
    "ClimateCompareEpochs": _climate_compare,
    "EnergyFacilityDistance": _energy_facility_distance,
    "EnergyMineralSiteDistance": _energy_mineral_distance,
    "EnergyBBoxArea": _energy_bbox,
    "EnergyNearestFacility": _energy_nearest,
    "EnergyCapacityValue": _energy_capacity,
    "EnergyNearestMineralSite": _energy_cross_nearest,
    "EnergyCountryRenewableSummary": _energy_mix,
    "EnergyCapacityGrowth": _energy_growth,
    "EnergyTransitionMetrics": _energy_transition,
    "EnergyRegionalOverlaySummary": _energy_overlay,
    "OceanMonthlyGridValue": _ocean_grid,
    "OceanProfileValue": _ocean_profile,
    "OceanCruiseQC": _ocean_qc,
    "OceanStationDistance": _ocean_station_distance,
    "OceanBBoxArea": _ocean_bbox,
    "OceanAnnualCarbonSummary": _ocean_annual,
    "OceanVerticalProfileDifference": _ocean_vertical,
    "OceanRegionalMapSummary": _ocean_map,
    "OceanSurfaceDeepMatch": _ocean_match,
    "OceanProfileDeduplication": _ocean_dedup,
    "DeepTimeDecimalToDMS": _decimal_to_dms,
    "DeepTimeHaversineDistance": _distance,
    "DeepTimeBearing": _bearing,
    "DeepTimeBBoxArea": _bbox_tool,
    "CCSDecimalToDMS": _decimal_to_dms,
    "CCSHaversineDistance": _distance,
    "CCSBearing": _bearing,
    "CCSBBoxArea": _bbox_tool,
    "CCSCheckCRSConsistency": _crs_check,
    "RemoteSensingListFiles": _rs_list,
    "RemoteSensingBatchImageMean": lambda p: _rs_batch_stat(p, "mean"),
    "RemoteSensingBatchHotspotPercentage": lambda p: _rs_batch_stat(p, "hotspot_percentage"),
    "RemoteSensingTiffAverage": _rs_tiff_average,
    "RemoteSensingBatchNDSI": _rs_batch_ndsi,
    "RemoteSensingNDWI": _rs_ndwi,
    "RemoteSensingNBR": _rs_nbr,
    "RemoteSensingCloudMask": _rs_cloud_mask,
    "RemoteSensingBatchHotspotRaster": _rs_hotspot_raster,
    "RemoteSensingNDVI": _rs_ndvi,
    "RemoteSensingNDTI": _rs_ndti,
    "RemoteSensingCountAboveThreshold": _rs_count_threshold,
    "RemoteSensingBatchImageSum": lambda p: _rs_batch_stat(p, "sum"),
    "RemoteSensingTiffDifference": _rs_difference,
    "RemoteSensingExtremeSnowLossPercentage": _rs_count_threshold,
    "RemoteSensingSubtract": lambda p: _rs_difference(p, True),
    "RemoteSensingFireIncreaseMap": _rs_fire_map,
    "NaturalHazardLoadEventTable": _hazard_load,
    "NaturalHazardFilterRecords": _hazard_filter,
    "NaturalHazardCountRecords": _hazard_count,
    "NaturalHazardSumField": _hazard_sum,
    "NaturalHazardGroupAggregate": _hazard_group,
    "NaturalHazardSelectMaxRecord": _hazard_max,
    "NaturalHazardBuildStartDate": _hazard_dates,
    "NaturalHazardSelectLatest": _hazard_latest,
}
_SCALAR_REMOTE = {
    "RemoteSensingMean",
    "RemoteSensingPercentageChange",
    "RemoteSensingDifference",
    "RemoteSensingDivision",
    "RemoteSensingLinearTrend",
    "RemoteSensingMaxValueAndIndex",
    "RemoteSensingKelvinToCelsius",
    "RemoteSensingMinValueAndIndex",
    "RemoteSensingSensSlope",
    "RemoteSensingCoefficientOfVariation",
    "RemoteSensingKurtosis",
    "RemoteSensingMannKendallTest",
    "RemoteSensingArgmax",
}


def dispatch(tool_name: str, input_str: str) -> str:
    params = parse_arguments(input_str)
    from agentgs.contracts.remote_sensing import RASTER_STEPS, VERSION, enabled, inspect_inputs

    rs_report = None
    if enabled() and tool_name in RASTER_STEPS and (params.get("_rs_profile") == VERSION):
        rs_report = inspect_inputs(
            tool_name, params, inspect_raster=True, resolve_path=_resolve_path
        )
        issues = list(params.get("_rs_binding_issues") or []) + rs_report["issues"]
        if issues:
            raise ValueError("Remote-sensing input contract: " + "; ".join(dict.fromkeys(issues)))
        if tool_name == "RemoteSensingCloudMask":
            sources = params.get("sr_band_paths") or [params.get("sr_band_path")]
            masks = params.get("qa_pixel_paths") or [params.get("qa_pixel_path")]
            for source, mask in zip(sources, masks):
                _require_matching_grids([_resolve_path(source), _resolve_path(mask)])
    if tool_name in _IMPLEMENTATIONS:
        result = _IMPLEMENTATIONS[tool_name](params)
        if rs_report is not None:
            payload = json.loads(result)
            payload["_rs_contract"] = rs_report
            return json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True)
        return result
    if tool_name in _SCALAR_REMOTE:
        return _rs_scalar(tool_name, params)
    for suffix, implementation in _GIS_SUFFIXES.items():
        if tool_name.endswith(suffix):
            return implementation(params)
    raise KeyError(f"No executable implementation registered for {tool_name}")


def _tool(tool_name: str) -> Callable[[str], str]:

    def run(input_str: str) -> str:
        try:
            params = parse_arguments(input_str)
            raw = dispatch(tool_name, input_str)
            output_keys = params.get("_output_keys") or []
            if output_keys:
                payload = json.loads(raw)
                standardized = payload.get("standardized_layers") or []
                reports = payload.get("validation_reports") or []
                standardized_paths = [
                    item.get("output")
                    for item in standardized
                    if isinstance(item, dict) and item.get("output")
                ]
                clean_paths = [
                    item.get("clean_output")
                    for item in reports
                    if isinstance(item, dict) and item.get("clean_output")
                ]
                for key in output_keys:
                    if key in payload:
                        continue
                    if key in ("standardized_points", "standardized_layer"):
                        payload[key] = standardized_paths[0] if standardized_paths else None
                    elif key == "standardized_layers":
                        payload[key] = standardized_paths
                    elif key in ("clean_points", "clean_leak_points", "leak_points_projected"):
                        payload[key] = (
                            clean_paths[0]
                            if clean_paths
                            else standardized_paths[0]
                            if standardized_paths
                            else None
                        )
                    elif key in ("clean_layers", "batch_layers", "joined_layers"):
                        payload[key] = clean_paths or standardized_paths
                    elif "report" in key:
                        payload[key] = reports
                    elif "neighbour" in key:
                        payload[key] = payload.get("nearest_neighbours")
                    elif "buffer" in key:
                        payload[key] = payload.get("buffer_output")
                    elif any(
                        (
                            token in key
                            for token in (
                                "overlay",
                                "join",
                                "zone_points",
                                "monitoring_zones",
                                "priority_zones",
                            )
                        )
                    ):
                        payload[key] = payload.get("overlay_output")
                    elif "raster" in key:
                        payload[key] = payload.get("raster_output")
                    elif "stats" in key:
                        payload[key] = {
                            name: value
                            for name, value in payload.items()
                            if name not in output_keys
                        }
                    elif key in ("map_png", "map_html"):
                        payload[key] = payload.get(key)
                    elif "dms" in key:
                        payload[key] = payload.get("dms")
                    elif "distance" in key:
                        payload[key] = payload.get("distance_km")
                    elif "bearing" in key:
                        payload[key] = payload.get("bearing_degrees")
                    elif "area" in key:
                        payload[key] = payload.get("area_km2")
                    else:
                        payload[key] = {
                            name: value
                            for name, value in payload.items()
                            if name not in output_keys
                        }
                return json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True)
            return raw
        except Exception as exc:
            return _result(
                tool=tool_name, status="error", error_type=type(exc).__name__, error=str(exc)
            )

    run.__name__ = tool_name
    run.__doc__ = f"Execute {tool_name} with a JSON string or documented compact text input."
    return run


def execute_domain(name, arguments):
    return _tool(name)(json.dumps(arguments, ensure_ascii=False))
