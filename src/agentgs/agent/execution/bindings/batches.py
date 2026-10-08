"""batches phase of the original parameter-binding policy."""

from __future__ import annotations

import re
from pathlib import Path

from agentgs.agent.execution.bindings.helpers import (
    _INVALID_VALUES,
    _partition_sequence,
    _semantic_period_partition,
    _source_raster_groups,
    _temporal_source_rasters,
)


def apply(state):
    state.file_list_tools = {
        "RemoteSensingBatchImageMean",
        "RemoteSensingBatchHotspotPercentage",
        "RemoteSensingTiffAverage",
        "RemoteSensingBatchHotspotRaster",
        "RemoteSensingBatchImageSum",
    }
    if state.tool_name in state.file_list_tools and (
        state.upstream_rasters or state.artifact_registry is not None
    ):
        authoritative_rasters = state.preferred_rasters or state.upstream_rasters
        cohort_already_selected = False
        if (
            state.tool_name == "RemoteSensingBatchImageMean"
            and state.occurrence_count > 1
            and (state.artifact_registry is not None)
        ):
            try:
                source_rasters = (
                    state.preferred_rasters
                    or state.artifact_registry.paths(
                        semantic_types=("Raster",), producer="RemoteSensingListFiles"
                    )
                    or state.artifact_registry.paths(
                        semantic_types=("Raster",), producer="query_resource"
                    )
                )
                if source_rasters:
                    authoritative_rasters = _semantic_period_partition(
                        source_rasters,
                        state.clean_question,
                        state.occurrence_index,
                        state.occurrence_count,
                    )
                    cohort_already_selected = True
            except Exception:
                pass
        if state.tool_name == "RemoteSensingTiffAverage" and state.artifact_registry is not None:
            try:
                source_rasters = state.preferred_rasters or state.artifact_registry.paths(
                    semantic_types=("Raster",), producer="query_resource"
                )
                if not source_rasters:
                    source_rasters = [
                        path
                        for path in state.raster_files
                        if "average_cohort_" not in Path(str(path)).name.casefold()
                    ]
                if source_rasters:
                    authoritative_rasters = _temporal_source_rasters(
                        source_rasters, state.occurrence_index, state.occurrence_count
                    )
            except Exception:
                pass
        if state.tool_name == "RemoteSensingBatchImageSum" and state.artifact_registry is not None:
            try:
                produced = [
                    str(record.value)
                    for record in state.artifact_registry.records
                    if record.semantic_type == "Raster"
                    and record.producer not in {"query_resource", "RemoteSensingListFiles"}
                ]
                if state.occurrence_index < len(produced):
                    authoritative_rasters = [produced[state.occurrence_index]]
                else:
                    source = state.artifact_registry.paths(
                        semantic_types=("Raster",), producer="query_resource"
                    )
                    groups = _source_raster_groups(source)
                    largest = max((len(items) for items in groups), default=0)
                    primary = next((items for items in groups if len(items) == largest), [])
                    residual = [item for item in source if item not in primary]
                    residual_groups = _source_raster_groups(residual)
                    residual_index = state.occurrence_index - len(produced)
                    if residual_groups:
                        authoritative_rasters = _partition_sequence(
                            residual, residual_index, max(1, state.occurrence_count - len(produced))
                        )
            except Exception:
                pass
        if (
            state.tool_name == "RemoteSensingBatchHotspotPercentage"
            and state.artifact_registry is not None
        ):
            lowered_question = state.clean_question.casefold()
            temporal_percentage = any(
                (
                    phrase in lowered_question
                    for phrase in (
                        "daily",
                        "each day",
                        "each sampling time",
                        "each time point",
                        "at each time",
                    )
                )
            )
            prior_tools = state.tool_chain[
                : max(0, int(state.execution_context.get("execution_index", 1) or 1) - 1)
            ]
            if "RemoteSensingTiffDifference" in prior_tools:
                try:
                    difference_sources = state.artifact_registry.paths(
                        semantic_types=("Raster",), producer="RemoteSensingTiffDifference"
                    )
                    difference_outputs = [
                        path
                        for path in difference_sources
                        if Path(str(path)).name.casefold().endswith("_difference.tif")
                    ]
                    if difference_outputs:
                        authoritative_rasters = difference_outputs
                except Exception:
                    pass
            elif temporal_percentage:
                try:
                    index_sources: list[str] = []
                    for producer in reversed(
                        (
                            "RemoteSensingNDVI",
                            "RemoteSensingNDWI",
                            "RemoteSensingNBR",
                            "RemoteSensingBatchNDSI",
                            "RemoteSensingNDTI",
                        )
                    ):
                        index_sources = state.artifact_registry.paths(
                            semantic_types=("Raster",), producer=producer
                        )
                        if index_sources:
                            break
                    if not index_sources and any(
                        (
                            marker in lowered_question
                            for marker in ("ndvi", "ndwi", "ndsi", "ndti", "nbr")
                        )
                    ):
                        index_sources = state.artifact_registry.paths(
                            semantic_types=("Raster",), producer="RemoteSensingListFiles"
                        ) or state.artifact_registry.paths(
                            semantic_types=("Raster",), producer="query_resource"
                        )
                    if index_sources:
                        authoritative_rasters = index_sources
                except Exception:
                    pass
        if (
            state.tool_name == "RemoteSensingTiffAverage" or cohort_already_selected
        ) and authoritative_rasters:
            state.params["file_list"] = list(authoritative_rasters)
        elif authoritative_rasters:
            state.params["file_list"] = _partition_sequence(
                authoritative_rasters, state.occurrence_index, state.occurrence_count
            )
    if (
        state.tool_name == "RemoteSensingBatchImageMean"
        and state.occurrence_count > 1
        and (state.params.get("output_path") in _INVALID_VALUES)
    ):
        state.params["output_path"] = f"batch_mean_cohort_{state.occurrence_index + 1}.tif"
    if (
        state.tool_name == "RemoteSensingTiffAverage"
        and state.occurrence_count > 1
        and (state.params.get("output_path") in _INVALID_VALUES)
    ):
        state.params["output_path"] = f"average_cohort_{state.occurrence_index + 1}.tif"
    if state.tool_name in {
        "RemoteSensingBatchHotspotPercentage",
        "RemoteSensingBatchHotspotRaster",
    }:
        lowered_question = state.clean_question.casefold()
        relative_match = re.search(
            "(\\d+(?:\\.\\d+)?)\\s*%\\s*(higher|above|greater|lower|below|less)\\s+than\\s+(?:the\\s+)?(?:mean|average)",
            lowered_question,
        )
        exceeds_mean_match = re.search(
            "(?:exceeds?|above|higher\\s+than)\\s+(?:the\\s+)?(?:[a-z0-9_-]+\\s+)?mean[^.]{0,30}?(\\d+(?:\\.\\d+)?)\\s*%",
            lowered_question,
        )
        average_threshold = any(
            (
                phrase in lowered_question
                for phrase in (
                    "above the average",
                    "above average",
                    "above the mean",
                    "higher than the mean",
                )
            )
        ) or bool(re.search("above\\s+(?:the\\s+)?(?:[a-z0-9_-]+\\s+)?mean", lowered_question))
        fixed_above = re.search(
            "(?:ndvi|ndwi|ndsi|index|value)\\s+(?:is\\s+)?(?:greater|higher|above|exceeds?)\\s+than\\s+([-+]?\\d+(?:\\.\\d+)?)",
            lowered_question,
        )
        loss_threshold = re.search(
            "(?:ndsi|ndwi|index)[^.]{0,45}?(?:decrease|drop|loss|drops?)[^.]{0,35}?(\\d+(?:\\.\\d+)?)\\s*(%)?",
            lowered_question,
        ) or re.search(
            "(?:decrease|drop|loss|drops?)[^.]{0,35}?(?:in\\s+)?(?:ndsi|ndwi|index)[^.]{0,35}?(\\d+(?:\\.\\d+)?)\\s*(%)?",
            lowered_question,
        )
        if loss_threshold:
            threshold = float(loss_threshold.group(1))
            if loss_threshold.group(2):
                threshold /= 100.0
            state.params["threshold"] = -abs(threshold)
            state.params["comparison"] = "below"
        elif fixed_above:
            state.params["threshold"] = float(fixed_above.group(1))
            state.params["comparison"] = "above"
        relative_percent = (
            float(relative_match.group(1))
            if relative_match
            else float(exceeds_mean_match.group(1))
            if exceeds_mean_match
            else 0.0
        )
        per_image_relative = bool(
            (relative_match or exceeds_mean_match)
            and any(
                (
                    phrase in lowered_question
                    for phrase in ("change map", "each change", "adjacent dates")
                )
            )
        )
        if per_image_relative:
            state.params["threshold"] = 0.0
            state.params["relative_to_each_mean"] = True
            state.params["relative_factor"] = 1.0 + relative_percent / 100.0
            state.params["comparison"] = "above"
        elif relative_match or exceeds_mean_match or average_threshold:
            scalar_means = [values[0] for values in state.numeric_series if len(values) == 1]
            cohort_series = [values for values in state.numeric_series if len(values) > 1]
            if state.occurrence_index < len(scalar_means):
                baseline = scalar_means[state.occurrence_index]
            elif scalar_means:
                baseline = scalar_means[-1]
            elif (
                state.occurrence_index < len(cohort_series)
                and cohort_series[state.occurrence_index]
            ):
                baseline = sum(cohort_series[state.occurrence_index]) / len(
                    cohort_series[state.occurrence_index]
                )
            else:
                baseline = None
            if baseline is not None:
                percent = relative_percent
                direction = relative_match.group(2) if relative_match else "above"
                factor = (
                    1.0 - percent / 100.0
                    if direction in {"lower", "below", "less"}
                    else 1.0 + percent / 100.0
                )
                state.params["threshold"] = float(baseline) * factor
                state.params["comparison"] = (
                    "below" if direction in {"lower", "below", "less"} else "above"
                )
        if state.tool_name == "RemoteSensingBatchHotspotRaster" and state.occurrence_count > 1:
            file_count = len(state.params.get("file_list") or [])
            state.params["output_path_list"] = [
                f"hotspot_cohort_{state.occurrence_index + 1}_{index + 1}.tif"
                for index in range(max(1, file_count))
            ]
