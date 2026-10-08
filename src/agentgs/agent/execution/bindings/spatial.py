"""spatial phase of the original parameter-binding policy."""

from __future__ import annotations

import re

from agentgs.agent.execution.bindings.helpers import (
    _INVALID_VALUES,
    _partition_sequence,
    _path_has_band,
    _role_band,
)


def apply(state):
    if state.tool_name == "RemoteSensingBatchNDSI" and (
        state.upstream_rasters or state.preferred_rasters
    ):
        index_rasters = state.preferred_rasters or state.upstream_rasters
        green_band = _role_band(state.clean_question, ("green",)) or 4
        swir_band = _role_band(state.clean_question, ("swir", "shortwave")) or 6
        green_files = [path for path in index_rasters if _path_has_band(path, green_band)]
        swir_files = [path for path in index_rasters if _path_has_band(path, swir_band)]
        pairs = state.pair_scene_files(green_files, swir_files)
        if pairs:
            selected_pairs = _partition_sequence(
                pairs, state.occurrence_index, state.occurrence_count
            )
            state.params["green_file_list"] = [first for first, _ in selected_pairs]
            state.params["swir_file_list"] = [second for _, second in selected_pairs]
    if state.tool_name.startswith("NaturalHazard"):
        sources = state.resource_params.get("sources")
        if isinstance(sources, list) and sources:
            state.params.setdefault("source", sources[0])
    if state.tool_name.endswith("DecimalToDMS"):
        axis = state.params.get("field") or state.params.get("axis")
        if not axis:
            axis = (
                "lon"
                if re.search("\\b(?:lon|longitude)\\b", state.clean_question, re.IGNORECASE)
                else "lat"
                if re.search("\\b(?:lat|latitude)\\b", state.clean_question, re.IGNORECASE)
                else None
            )
        degree_match = re.search("(-?\\d+(?:\\.\\d+)?)\\s*°", state.clean_question)
        if degree_match:
            state.params["value"] = float(degree_match.group(1))
        if axis:
            state.params["field"] = axis
    if state.tool_name.endswith("BBoxArea") and state.params.get("bbox") in _INVALID_VALUES:
        bbox_match = re.search(
            "(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)",
            state.clean_question,
        )
        if bbox_match:
            state.params["bbox"] = [float(value) for value in bbox_match.groups()]
    spatial_workflow = any(
        (
            token in state.tool_name
            for token in (
                "Spatial",
                "NearestNeighbor",
                "Buffer",
                "Overlay",
                "BatchVector",
                "InterpolateIDW",
                "RasterAlgebra",
                "ZonalStatistics",
                "RenderMap",
            )
        )
    )
    if spatial_workflow:
        state.params["paths"] = list(dict.fromkeys(state.output_paths + state.source_paths))
    if state.tool_name.endswith("NearestNeighbor"):
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        nearest_match = re.search(
            "\\b(one|two|three|four|five|\\d+)\\s*[- ]?nearest[- ]?neighbou?rs?\\b",
            state.clean_question,
            re.IGNORECASE,
        )
        if nearest_match:
            raw_count = nearest_match.group(1).casefold()
            state.params["k"] = word_numbers.get(
                raw_count, int(raw_count) if raw_count.isdigit() else 1
            )
    if state.tool_name.endswith("Buffer"):
        multi_ring = re.search(
            "(\\d+(?:\\.\\d+)?)\\s*[,，]\\s*(\\d+(?:\\.\\d+)?)\\s*[,，]\\s*(\\d+(?:\\.\\d+)?)\\s*(?:km|kilomet(?:er|re)s?)",
            state.clean_question,
            re.IGNORECASE,
        )
        if multi_ring:
            state.params["buffer_distances_km"] = [float(value) for value in multi_ring.groups()]
        else:
            buffer_match = re.search(
                "(\\d+(?:\\.\\d+)?)\\s*[- ]?(?:km|kilomet(?:er|re)s?)\\s+buffers?",
                state.clean_question,
                re.IGNORECASE,
            )
            if buffer_match:
                state.params["distance_km"] = float(buffer_match.group(1))
    if state.tool_name.endswith("RenderMap"):
        field_rules = (
            (
                ("net stored", "net-stored", "co2_net_stored_tonnes", "净封存"),
                "co2_net_stored_tonnes",
            ),
            (("injection", "注入"), "co2_injected_tonnes"),
            (("reservoir pressure", "pressure", "压力"), "avg_reservoir_pressure_MPa"),
            (("abundance", "丰度"), "abundance"),
            (("geological age", "by age", "age field", "年龄"), "age"),
            (("zone", "分区"), "zone_id"),
        )
        lowered_question = state.clean_question.casefold()
        for phrases, field in field_rules:
            if any((phrase.casefold() in lowered_question for phrase in phrases)):
                state.params["classification_field"] = field
                break
        state.params.setdefault("title", state.clean_question[:120] or "GeoToolAgent map output")
        if "html" in lowered_question and "png" in lowered_question:
            state.params["output_formats"] = ["PNG", "HTML"]
