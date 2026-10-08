"""initialize phase of the original parameter-binding policy."""

from __future__ import annotations

from agentgs.agent.execution.bindings.helpers import (
    _INVALID_VALUES,
    _json_objects,
    _named_query_parameters,
    _numeric_series,
    _parse_generated,
    _paths_from,
    _raster_files,
    _resource_object,
    split_question_resource,
)


def apply(state):
    state.clean_question, state.resource = split_question_resource(state.question)
    state.params = _parse_generated(state.generated_input)
    state.resource_params = _resource_object(state.resource)
    for key, value in _named_query_parameters(state.clean_question).items():
        if state.params.get(key) in _INVALID_VALUES:
            state.params[key] = value
    for key, value in state.resource_params.items():
        if key not in ("sources",):
            state.params[key] = value
    if isinstance(state.resource, str):
        state.params.setdefault("input_resource", state.resource)
    state.current_objects = _json_objects(state.current_output)
    state.output_objects = _json_objects(state.previous_output)
    for current_object in state.current_objects:
        if current_object not in state.output_objects:
            state.output_objects.append(current_object)
    state.output_paths = _paths_from(state.output_objects)
    state.source_paths = _paths_from([state.resource_params])
    state.raster_files = _raster_files(state.output_objects)
    state.current_raster_files = _raster_files(state.current_objects)
    state.numeric_series = _numeric_series(state.output_objects)
    state.execution_context = state.execution_context or {}
    state.occurrence_index = int(state.execution_context.get("occurrence_index", 0) or 0)
    state.occurrence_count = max(1, int(state.execution_context.get("occurrence_count", 1) or 1))
    state.tool_chain = list(state.execution_context.get("tool_chain") or [])
    state.preferred_rasters: list[str] = []
    if state.artifact_registry is not None:
        try:
            registry_paths = state.artifact_registry.paths(
                semantic_types=("Raster", "Vector", "Table", "File", "Collection")
            )
            state.output_paths = list(dict.fromkeys(state.output_paths + registry_paths))
            state.raster_files = list(
                dict.fromkeys(
                    state.raster_files + state.artifact_registry.paths(semantic_types=("Raster",))
                )
            )
            state.current_raster_files = list(
                dict.fromkeys(
                    state.current_raster_files
                    + state.artifact_registry.paths(
                        semantic_types=("Raster",), producer=state.tool_name
                    )
                )
            )
            state.execution_index = int(state.execution_context.get("execution_index", 1) or 1)
            for producer in reversed(state.tool_chain[: max(0, state.execution_index - 1)]):
                if producer == state.tool_name:
                    continue
                candidates = state.artifact_registry.paths(
                    semantic_types=("Raster",), producer=producer
                )
                if candidates:
                    state.preferred_rasters = candidates
                    break
            if not state.preferred_rasters:
                state.preferred_rasters = state.artifact_registry.paths(
                    semantic_types=("Raster",), producer="query_resource"
                )
        except Exception:
            pass
