"""series phase of the original parameter-binding policy."""

from __future__ import annotations

from agentgs.agent.execution.bindings.helpers import (
    _artifact_producer_series,
    _select_numeric_series,
    _set_if_missing,
)


def apply(state):
    state.scalar_x_tools = {
        "RemoteSensingMean",
        "RemoteSensingMaxValueAndIndex",
        "RemoteSensingMinValueAndIndex",
        "RemoteSensingSensSlope",
        "RemoteSensingCoefficientOfVariation",
        "RemoteSensingKurtosis",
        "RemoteSensingMannKendallTest",
        "RemoteSensingArgmax",
    }
    if state.tool_name in state.scalar_x_tools and state.numeric_series:
        if state.tool_name in {
            "RemoteSensingMaxValueAndIndex",
            "RemoteSensingMinValueAndIndex",
            "RemoteSensingArgmax",
        }:
            pct_values: list[float] = []
            if state.artifact_registry is not None:
                try:
                    pct_values = [
                        float(record.value)
                        for record in state.artifact_registry.records
                        if record.producer == "RemoteSensingPercentageChange"
                        and isinstance(record.value, (int, float))
                    ]
                except Exception:
                    pct_values = []
            percentage_steps_before = sum(
                (
                    1
                    for name in state.tool_chain[
                        : max(0, int(state.execution_context.get("execution_index", 1) or 1) - 1)
                    ]
                    if name == "RemoteSensingPercentageChange"
                )
            )
            if len(pct_values) >= 2:
                selected_series = pct_values
            elif percentage_steps_before >= 1:
                source_series = max(state.numeric_series, key=len, default=None)
                if source_series and len(source_series) >= 2:
                    selected_series = [
                        (source_series[index + 1] - source_series[index])
                        / source_series[index]
                        * 100
                        if source_series[index] != 0
                        else float("nan")
                        for index in range(len(source_series) - 1)
                    ]
            else:
                hotspot_series: list[list[float]] = []
                for obj in state.current_objects + state.output_objects:
                    if obj.get("statistic") != "hotspot_percentage":
                        continue
                    values = obj.get("values")
                    if isinstance(values, list):
                        series = [
                            float(item["value"])
                            for item in values
                            if isinstance(item, dict)
                            and isinstance(item.get("value"), (int, float))
                        ]
                        if series:
                            hotspot_series.append(series)
                asks_for_proportion_extremum = any(
                    (
                        phrase in state.clean_question.casefold()
                        for phrase in (
                            "largest proportion",
                            "highest proportion",
                            "most severe water loss",
                            "highest ndwi",
                        )
                    )
                )
                if asks_for_proportion_extremum and hotspot_series:
                    selected_series = hotspot_series[-1]
                else:
                    selected_series = max(state.numeric_series, key=len, default=None)
        elif state.tool_name == "RemoteSensingMean" and state.occurrence_count > 1:
            selected_series = None
            if state.artifact_registry is not None and state.tool_chain:
                state.execution_index = int(state.execution_context.get("execution_index", 1) or 1)
                preceding_tools = state.tool_chain[: max(0, state.execution_index - 1)]
                for producer in reversed(preceding_tools):
                    if producer == state.tool_name:
                        continue
                    produced_series = _artifact_producer_series(
                        state.artifact_registry, producer=producer
                    )
                    if len(produced_series) >= state.occurrence_count:
                        selected_series = _select_numeric_series(
                            produced_series, state.occurrence_index, state.occurrence_count
                        )
                        break
            cohorts = [series for series in state.numeric_series if len(series) > 1]
            if selected_series is None:
                selected_series = _select_numeric_series(
                    cohorts, state.occurrence_index, state.occurrence_count
                )
            if selected_series is None:
                scalar_cohorts = [series for series in state.numeric_series if len(series) == 1]
                selected_series = _select_numeric_series(
                    scalar_cohorts, state.occurrence_index, state.occurrence_count
                )
        elif state.tool_name in {
            "RemoteSensingSensSlope",
            "RemoteSensingMannKendallTest",
            "RemoteSensingCoefficientOfVariation",
            "RemoteSensingKurtosis",
        }:
            cohorts = [series for series in state.numeric_series if len(series) > 1]
            selected_series = _select_numeric_series(
                cohorts, state.occurrence_index, state.occurrence_count
            )
        else:
            selected_series = _select_numeric_series(
                state.numeric_series, state.occurrence_index, state.occurrence_count
            )
        if selected_series is not None:
            state.params["x"] = selected_series
    if state.tool_name == "RemoteSensingLinearTrend" and state.numeric_series:
        trend_series = None
        if state.artifact_registry is not None and state.tool_chain:
            state.execution_index = int(state.execution_context.get("execution_index", 1) or 1)
            for producer in reversed(state.tool_chain[: max(0, state.execution_index - 1)]):
                if producer == state.tool_name:
                    continue
                produced_series = _artifact_producer_series(
                    state.artifact_registry, producer=producer
                )
                if produced_series:
                    trend_series = max(produced_series, key=len)
                    break
        _set_if_missing(state.params, "y", trend_series or max(state.numeric_series, key=len))
    if state.tool_name == "RemoteSensingKelvinToCelsius" and state.numeric_series:
        scalar_series = [series for series in state.numeric_series if len(series) == 1]
        if scalar_series:
            _set_if_missing(state.params, "kelvin", scalar_series[-1][0])
