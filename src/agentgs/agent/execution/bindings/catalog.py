"""catalog phase of the original parameter-binding policy."""

from __future__ import annotations

import re

from agentgs.agent.execution.bindings.helpers import (
    _INVALID_VALUES,
    _set_if_missing,
)


def apply(state):
    if state.tool_name == "GeoBigDataExecuteGeoSPARQL":
        for forbidden in (
            "question",
            "exact_question",
            "exact question",
            "gold_sparql",
            "gold_answer",
            "endpoint",
        ):
            state.params.pop(forbidden, None)
        if isinstance(state.resource, str):
            state.params.setdefault("dataset", state.resource)
        if "sparql" in state.params and "query" not in state.params:
            state.params["query"] = state.params.pop("sparql")
    if state.tool_name == "DeepEarthQueryLITHO1":
        lowered_question = state.clean_question.casefold()
        if any(
            (
                token in lowered_question
                for token in ("lid-bottom", "lithospheric lid", "lid base", "lid bottom")
            )
        ):
            state.params["layer"] = "lid"
        elif "moho" in lowered_question:
            state.params["layer"] = "moho"
    if state.tool_name == "DeepEarthQueryUSGSCatalog":
        lowered_question = state.clean_question.casefold()
        if any(
            (
                token in lowered_question
                for token in ("largest magnitude", "maximum magnitude", "highest magnitude")
            )
        ):
            state.params["operation"] = "largest"
        elif any((token in lowered_question for token in ("deepest", "maximum depth"))):
            state.params["operation"] = "deepest"
        elif re.search(
            "\\b(?:total|sum(?:med)?|aggregate[sd]?)\\b[^.\\n]{0,40}\\benergy\\b|\\benergy\\b[^.\\n]{0,40}\\b(?:total|sum(?:med)?|aggregate[sd]?)\\b",
            lowered_question,
        ):
            state.params["operation"] = "total_energy"
        elif state.params.get("event_id"):
            state.params["operation"] = "event"
    if isinstance(state.resource, str):
        if state.tool_name == "MineralResourcesRankCommodityCountries":
            state.params["commodity"] = state.resource
        elif state.tool_name == "MineralResourcesCountrySummary":
            state.params["country"] = state.resource
        elif state.tool_name == "MineralResourcesGetDeposit" and state.resource.strip().isdigit():
            state.params["dep_id"] = int(state.resource)
    elif (
        state.tool_name == "MineralResourcesGetDeposit"
        and isinstance(state.resource, (int, float))
        and (not isinstance(state.resource, bool))
    ):
        state.params["dep_id"] = int(state.resource)
    if state.tool_name == "ClimateQueryFutureSSP":
        _set_if_missing(state.params, "scenario", state.resource_params.get("ssp"))
        _set_if_missing(state.params, "horizon_year", state.resource_params.get("ssp_horizon"))
        years = [
            int(value) for value in re.findall("\\b(?:19|20|21)\\d{2}\\b", state.clean_question)
        ]
        if (
            state.resource_params.get("ssp_horizon") in _INVALID_VALUES
            and (not years)
            and (state.params.get("scenario") not in _INVALID_VALUES)
        ):
            state.params["horizon_year"] = 2100
        elif state.params.get("horizon_year") in _INVALID_VALUES and years:
            state.params["horizon_year"] = years[-1]
    if state.tool_name == "ClimateQueryDeepTime":
        requested_ages = state.resource_params.get("ages_kyr_bp")
        if isinstance(requested_ages, list) and requested_ages and (state.occurrence_count > 1):
            state.params.pop("ages_kyr_bp", None)
            state.params["age_kyr_bp"] = requested_ages[
                min(state.occurrence_index, len(requested_ages) - 1)
            ]
        else:
            _set_if_missing(
                state.params, "age_kyr_bp", state.resource_params.get("deep_age_kyr_bp")
            )
        lowered_question = state.clean_question.casefold()
        if state.params.get("dataset") in _INVALID_VALUES:
            if "bintanja" in lowered_question:
                state.params["dataset"] = "Bintanja2005"
            elif "friedrich" in lowered_question:
                state.params["dataset"] = "Friedrich2016"
    if (
        state.tool_name == "ClimateQueryPresentMean"
        and state.resource_params.get("modern_year") is not None
    ):
        state.params.setdefault("year_start", state.resource_params["modern_year"])
        state.params.setdefault("year_end", state.resource_params["modern_year"])
