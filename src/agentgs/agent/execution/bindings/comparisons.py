"""comparisons phase of the original parameter-binding policy."""

from __future__ import annotations

import re

from agentgs.agent.execution.bindings.helpers import (
    _artifact_producer_series,
    _chronological_primary_rasters,
    _named_value_series,
    _partition_sequence,
    _select_numeric_series,
    _set_if_missing,
)


def apply(state):
    state.rs_auto_consecutive_batch = False
    if state.tool_name in {"RemoteSensingTiffDifference", "RemoteSensingSubtract"}:
        candidates = state.preferred_rasters or state.current_raster_files or state.raster_files
        if len(candidates) >= 2:
            if state.tool_name == "RemoteSensingTiffDifference" and any(
                (
                    phrase in state.clean_question.casefold()
                    for phrase in ("adjacent dates", "consecutive dates", "between consecutive")
                )
            ):
                source = candidates
                if state.artifact_registry is not None:
                    try:
                        source = (
                            state.artifact_registry.paths(
                                semantic_types=("Raster",), producer="RemoteSensingListFiles"
                            )
                            or state.artifact_registry.paths(
                                semantic_types=("Raster",), producer="query_resource"
                            )
                            or candidates
                        )
                    except Exception:
                        source = candidates
                ordered = _chronological_primary_rasters(list(source))
                if len(ordered) >= 2:
                    state.params["image_a_paths"] = ordered[1:]
                    state.params["image_b_paths"] = ordered[:-1]
                    state.rs_auto_consecutive_batch = True
            if state.occurrence_count > 1 and len(candidates) > state.occurrence_index + 1:
                first, second = (
                    candidates[state.occurrence_index],
                    candidates[state.occurrence_index + 1],
                )
            else:
                first, second = (candidates[-2], candidates[-1])
            if state.tool_name == "RemoteSensingTiffDifference":
                state.params["image_a_path"], state.params["image_b_path"] = (first, second)
            else:
                lowered_question = state.clean_question.casefold()
                if any((token in lowered_question for token in ("increase", "growth", "gain"))):
                    first, second = (second, first)
                state.params["img1_path"], state.params["img2_path"] = (first, second)
                if any(
                    (
                        token in lowered_question
                        for token in ("maxfrp", "fire intensity", "fire-prone")
                    )
                ):
                    state.params["nodata_as_zero"] = True
    if state.tool_name == "RemoteSensingFireIncreaseMap":
        candidates = state.preferred_rasters or state.current_raster_files or state.raster_files
        if candidates:
            state.params["change_image_path"] = candidates[-1]
        threshold_match = re.search(
            "(?:\\+|above|greater than|threshold(?:\\s+of)?)\\s*(\\d+(?:\\.\\d+)?)\\s*mw",
            state.clean_question,
            re.IGNORECASE,
        )
        if threshold_match:
            state.params["threshold"] = float(threshold_match.group(1))
    if state.tool_name in {
        "RemoteSensingCountAboveThreshold",
        "RemoteSensingExtremeSnowLossPercentage",
    }:
        candidates = state.preferred_rasters or state.current_raster_files or state.raster_files
        if candidates:
            key = (
                "file_path"
                if state.tool_name == "RemoteSensingCountAboveThreshold"
                else "binary_map_path"
            )
            if state.occurrence_count > 1:
                selected = _partition_sequence(
                    candidates, state.occurrence_index, state.occurrence_count
                )
                state.params[key] = selected[0] if selected else candidates[-1]
            else:
                state.params[key] = candidates[-1]
            state.params.setdefault("threshold", 0.5)
    if state.tool_name == "ClimateCompareEpochs" and state.numeric_series:
        flattened = [value for values in state.numeric_series for value in values]
        if len(flattened) >= 2:
            _set_if_missing(state.params, "values", flattened[-3:])
    if state.tool_name in {
        "RemoteSensingDifference",
        "RemoteSensingPercentageChange",
        "RemoteSensingDivision",
    }:
        artifact_pair: tuple[float, float] | None = None
        if (
            state.tool_name in {"RemoteSensingDifference", "RemoteSensingPercentageChange"}
            and state.artifact_registry is not None
            and state.tool_chain
        ):
            state.execution_index = int(state.execution_context.get("execution_index", 1) or 1)
            preceding_tools = state.tool_chain[: max(0, state.execution_index - 1)]
            for producer in reversed(preceding_tools):
                if producer == state.tool_name:
                    continue
                if (
                    state.tool_name == "RemoteSensingPercentageChange"
                    and producer == "RemoteSensingDifference"
                ):
                    continue
                produced_series = _artifact_producer_series(
                    state.artifact_registry, producer=producer
                )
                if len(produced_series) < 2:
                    continue
                reduced = [sum(series) / len(series) for series in produced_series[-2:] if series]
                if len(reduced) == 2:
                    artifact_pair = (float(reduced[0]), float(reduced[1]))
                    break
        scalars = [values[0] for values in state.numeric_series if len(values) == 1]
        if artifact_pair is not None:
            state.params["a"], state.params["b"] = artifact_pair
        elif len(scalars) >= 2:
            difference_precedes_current = (
                state.tool_name == "RemoteSensingPercentageChange"
                and "RemoteSensingDifference"
                in state.tool_chain[
                    : max(0, int(state.execution_context.get("execution_index", 1) or 1) - 1)
                ]
            )
            if difference_precedes_current and len(scalars) >= 3:
                state.params["a"] = scalars[-3]
                state.params["b"] = scalars[-2]
            else:
                state.params["a"] = scalars[-2]
                state.params["b"] = scalars[-1]
        elif len([values for values in state.numeric_series if len(values) > 1]) >= 2:
            cohorts = [values for values in state.numeric_series if len(values) > 1]
            state.params["a"] = sum(cohorts[-2]) / len(cohorts[-2])
            state.params["b"] = sum(cohorts[-1]) / len(cohorts[-1])
        else:
            scalar_matches = re.findall(
                "\\\\?\\{?\\s*\\\\?[\\\"']value\\\\?[\\\"']\\s*:\\s*([-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)\\s*\\\\?\\}",
                f"{state.previous_output}\n{state.current_output}",
            )
            if len(scalar_matches) >= 2:
                state.params["a"] = float(scalar_matches[-2])
                state.params["b"] = float(scalar_matches[-1])
    if state.tool_name == "RemoteSensingDifference":
        hotspot_series: list[list[float]] = []
        for obj in state.current_objects + state.output_objects:
            if obj.get("statistic") != "hotspot_percentage":
                continue
            values = obj.get("values")
            if not isinstance(values, list):
                continue
            series = [
                float(item["value"])
                for item in values
                if isinstance(item, dict) and isinstance(item.get("value"), (int, float))
            ]
            if series:
                hotspot_series.append(series)
        if len(hotspot_series) >= 2:
            state.params["a"] = sum(hotspot_series[-2]) / len(hotspot_series[-2])
            state.params["b"] = sum(hotspot_series[-1]) / len(hotspot_series[-1])
    if (
        state.tool_name in {"RemoteSensingDifference", "RemoteSensingPercentageChange"}
        and state.occurrence_count > 1
    ):
        if state.tool_name == "RemoteSensingPercentageChange":
            selected = max(state.numeric_series, key=len, default=None)
        else:
            selected = _select_numeric_series(
                state.numeric_series, state.occurrence_index, state.occurrence_count
            )
        if selected and len(selected) >= 2:
            if (
                state.tool_name == "RemoteSensingPercentageChange"
                and len(selected) > state.occurrence_index + 1
            ):
                state.params["a"], state.params["b"] = (
                    selected[state.occurrence_index],
                    selected[state.occurrence_index + 1],
                )
            else:
                state.params["a"], state.params["b"] = (selected[-2], selected[-1])
    if state.tool_name == "RemoteSensingDivision":
        if state.artifact_registry is not None and state.occurrence_count > 1:
            try:
                sum_values = [
                    float(values[0]) for values in state.numeric_series if len(values) == 1
                ]
                if len(sum_values) < state.occurrence_count * 2:
                    sum_values = [
                        float(record.value)
                        for record in state.artifact_registry.records
                        if record.producer == "RemoteSensingBatchImageSum"
                        and isinstance(record.value, (int, float))
                    ]
                if len(sum_values) >= state.occurrence_count * 2:
                    state.params["a"] = sum_values[state.occurrence_index]
                    state.params["b"] = sum_values[state.occurrence_count + state.occurrence_index]
            except Exception:
                pass
        ratio_match = re.search(
            "ratio\\s+of\\s+([a-z0-9_]+)\\s+to\\s+([a-z0-9_]+)", state.clean_question, re.IGNORECASE
        )
        if ratio_match:
            numerator = _named_value_series(state.output_objects, ratio_match.group(1))
            denominator = _named_value_series(state.output_objects, ratio_match.group(2))
            if numerator and len(numerator) == len(denominator):
                state.params["a_values"] = numerator
                state.params["b_values"] = denominator
