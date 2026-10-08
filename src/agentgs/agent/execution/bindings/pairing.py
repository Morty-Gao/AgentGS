"""pairing phase of the original parameter-binding policy."""

from __future__ import annotations

from pathlib import Path

from agentgs.agent.execution.bindings.helpers import (
    _band_numbers,
    _pair_scene_files,
    _parse_generated,
    _partition_sequence,
    _path_has_band,
    _role_band,
)


def apply(state):
    if state.tool_name == "RemoteSensingListFiles":
        if isinstance(state.resource, str):
            if state.occurrence_index > 0 and state.output_paths:
                state.params["dir_path"] = str(Path(state.output_paths[-1]).parent)
            else:
                state.params["dir_path"] = state.resource
        elif state.source_paths:
            state.params.setdefault("dir_path", str(Path(state.source_paths[0]).parent))
    from agentgs.contracts.remote_sensing import RASTER_STEPS, align_scene_pairs
    from agentgs.contracts.remote_sensing import enabled as rs_enabled

    state.rs_pair_issues = []
    state.rs_active = (
        rs_enabled()
        and state.tool_name in RASTER_STEPS
        and (state.execution_context.get("remote_sensing_contracts") is not False)
    )

    def pair_scene_files(first, second):
        if not state.rs_active:
            return _pair_scene_files(first, second)
        try:
            return align_scene_pairs(
                first, second, reuse_qa=state.tool_name == "RemoteSensingCloudMask"
            )
        except ValueError as exc:
            from agentgs.contracts.remote_sensing import inspect_inputs

            explicit = _parse_generated(state.generated_input)
            check = inspect_inputs(state.tool_name, explicit, inspect_raster=True)
            if check["issues"]:
                state.rs_pair_issues.append(str(exc))
            return []

    state.pair_scene_files = pair_scene_files
    state.upstream_rasters = state.current_raster_files or state.raster_files
    if state.tool_name == "RemoteSensingCloudMask" and state.upstream_rasters:
        qa_files = [
            path for path in state.upstream_rasters if "qa_pixel" in Path(path).name.casefold()
        ]
        source_files = [path for path in state.upstream_rasters if path not in qa_files]
        requested_bands = _band_numbers(state.clean_question)
        if requested_bands:
            band_files = [
                path
                for path in source_files
                if any((_path_has_band(path, band) for band in requested_bands))
            ]
            if band_files:
                source_files = band_files
        pairs = state.pair_scene_files(source_files, qa_files)
        if pairs:
            selected_pairs = _partition_sequence(
                pairs, state.occurrence_index, state.occurrence_count
            )
            state.params["sr_band_paths"] = [first for first, _ in selected_pairs]
            state.params["qa_pixel_paths"] = [second for _, second in selected_pairs]
    if state.tool_name == "RemoteSensingNDWI" and (
        state.upstream_rasters or state.preferred_rasters
    ):
        index_rasters = state.preferred_rasters or state.upstream_rasters
        mentioned_bands = sorted(_band_numbers(state.clean_question))
        green_band = _role_band(state.clean_question, ("green",))
        nir_band = _role_band(state.clean_question, ("near-infrared", "near infrared", "nir"))
        if green_band is None and mentioned_bands:
            green_band = mentioned_bands[0]
        if nir_band is None and len(mentioned_bands) >= 2:
            nir_band = mentioned_bands[-1]
        green_band = green_band or 3
        nir_band = nir_band or 5
        green_files = [path for path in index_rasters if _path_has_band(path, green_band)]
        nir_files = [path for path in index_rasters if _path_has_band(path, nir_band)]
        pairs = state.pair_scene_files(green_files, nir_files)
        if pairs:
            selected_pairs = _partition_sequence(
                pairs, state.occurrence_index, state.occurrence_count
            )
            state.params["input_swir_paths"] = [first for first, _ in selected_pairs]
            state.params["input_nir_paths"] = [second for _, second in selected_pairs]
    if state.tool_name == "RemoteSensingNDVI" and (
        state.upstream_rasters or state.preferred_rasters
    ):
        index_rasters = state.preferred_rasters or state.upstream_rasters
        mentioned_bands = sorted(_band_numbers(state.clean_question))
        red_band = _role_band(state.clean_question, ("red",))
        nir_band = _role_band(state.clean_question, ("near-infrared", "near infrared", "nir"))
        if red_band is None and mentioned_bands:
            red_band = mentioned_bands[0]
        if nir_band is None and len(mentioned_bands) >= 2:
            nir_band = mentioned_bands[-1]
        if red_band is None or nir_band is None:
            available_bands = sorted(
                {band for path in index_rasters for band in _band_numbers(Path(path).name)}
            )
            if len(available_bands) >= 2:
                red_band = red_band or available_bands[0]
                nir_band = nir_band or available_bands[-1]
        if red_band is not None and nir_band is not None and (red_band != nir_band):
            red_files = [path for path in index_rasters if _path_has_band(path, red_band)]
            nir_files = [path for path in index_rasters if _path_has_band(path, nir_band)]
            pairs = state.pair_scene_files(nir_files, red_files)
            if pairs:
                selected_pairs = _partition_sequence(
                    pairs, state.occurrence_index, state.occurrence_count
                )
                state.params["nir_band_files"] = [first for first, _ in selected_pairs]
                state.params["red_band_files"] = [second for _, second in selected_pairs]
                state.params["nir_band_file"] = selected_pairs[0][0]
                state.params["red_band_file"] = selected_pairs[0][1]
    if state.tool_name == "RemoteSensingNBR" and (
        state.upstream_rasters or state.preferred_rasters
    ):
        index_rasters = state.preferred_rasters or state.upstream_rasters
        mentioned_bands = sorted(_band_numbers(state.clean_question))
        nir_band = _role_band(state.clean_question, ("near-infrared", "near infrared", "nir"))
        swir_band = _role_band(state.clean_question, ("shortwave", "swir"))
        if nir_band is None and mentioned_bands:
            nir_band = mentioned_bands[0]
        if swir_band is None and len(mentioned_bands) >= 2:
            swir_band = mentioned_bands[-1]
        if nir_band is not None and swir_band is not None and (nir_band != swir_band):
            nir_files = [path for path in index_rasters if _path_has_band(path, nir_band)]
            swir_files = [path for path in index_rasters if _path_has_band(path, swir_band)]
            pairs = state.pair_scene_files(nir_files, swir_files)
            if pairs:
                selected_pairs = _partition_sequence(
                    pairs, state.occurrence_index, state.occurrence_count
                )
                state.params["input_nir_paths"] = [first for first, _ in selected_pairs]
                state.params["input_swir_paths"] = [second for _, second in selected_pairs]
    if state.tool_name == "RemoteSensingNDTI" and (
        state.upstream_rasters or state.preferred_rasters
    ):
        index_rasters = state.preferred_rasters or state.upstream_rasters
        mentioned_bands = sorted(_band_numbers(state.clean_question))
        if len(mentioned_bands) >= 2:
            green_band = _role_band(state.clean_question, ("green",)) or mentioned_bands[0]
            red_band = _role_band(state.clean_question, ("red",)) or mentioned_bands[-1]
            green_files = [path for path in index_rasters if _path_has_band(path, green_band)]
            red_files = [path for path in index_rasters if _path_has_band(path, red_band)]
            pairs = state.pair_scene_files(green_files, red_files)
            if pairs:
                state.params["input_pairs"] = _partition_sequence(
                    [{"input_green_path": green, "input_red_path": red} for green, red in pairs],
                    state.occurrence_index,
                    state.occurrence_count,
                )
