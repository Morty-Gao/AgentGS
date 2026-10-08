"""Ordered domain binding pipeline; only the declared GeoQuestions protocol differs."""

from types import SimpleNamespace

from agentgs.agent.execution.bindings import (
    batches,
    catalog,
    comparisons,
    finalize,
    initialize,
    pairing,
    series,
    spatial,
)
from agentgs.agent.execution.bindings.helpers import (
    _parse_generated as _parse_generated,
)
from agentgs.agent.execution.bindings.helpers import (
    split_question_resource as split_question_resource,
)
from agentgs.agent.execution.bindings.helpers import (
    tool_input_is_ready as _legacy_ready,
)


def prepare_tool_input(
    tool_name,
    generated_input,
    question,
    current_output="",
    previous_output="",
    artifact_registry=None,
    execution_context=None,
):
    state = SimpleNamespace(
        tool_name=tool_name,
        generated_input=generated_input,
        question=question,
        current_output=current_output,
        previous_output=previous_output,
        artifact_registry=artifact_registry,
        execution_context=execution_context,
    )
    for phase in [initialize, catalog, pairing, spatial, batches, series, comparisons]:
        result = phase.apply(state)
        if result is not None:
            return result
    return finalize.apply(state)


def tool_input_is_ready(tool_name, normalized_input):
    if tool_name == "GeoBigDataExecuteGeoSPARQL":
        params = _parse_generated(normalized_input)
        return bool(params.get("source_id") or params.get("query") or params.get("sparql"))
    return _legacy_ready(tool_name, normalized_input)
