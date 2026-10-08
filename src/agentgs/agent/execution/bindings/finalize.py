"""finalize phase of the original parameter-binding policy."""

from __future__ import annotations

import json
import re


def apply(state):
    from agentgs.contracts.remote_sensing import prepare_step

    if state.tool_name == "GeoBigDataExecuteGeoSPARQL":
        # The authorized snapshot interface receives a locator, not generic
        # resource annotations or live-endpoint parameters.
        state.params = {
            key: value
            for key, value in state.params.items()
            if key in {"source_id", "query", "dataset"}
        }

    rs_report = prepare_step(
        state.tool_name,
        state.params,
        {
            **state.execution_context,
            "rs_auto_consecutive_batch": state.rs_auto_consecutive_batch,
            "rs_requires_composite": bool(
                re.search(
                    "\\b(?:average|mean)\\s+(?:raster|image|map)\\b|\\bcomposit\\w*\\b|平均图|合成图",
                    state.clean_question,
                    re.I,
                )
            ),
        },
    )
    if rs_report and state.rs_pair_issues:
        rs_report["issues"].extend(state.rs_pair_issues)
        state.params["_rs_binding_issues"] = rs_report["issues"]
    repaired = json.dumps(state.params, ensure_ascii=False, separators=(",", ":"))
    metadata = {
        "type": "tool_input_normalization",
        "tool_name": state.tool_name,
        "generated_input": state.generated_input,
        "normalized_input": repaired,
        "authoritative_resource_applied": state.resource is not None,
        "upstream_object_count": len(state.output_objects),
        "occurrence_index": state.occurrence_index,
        "occurrence_count": state.occurrence_count,
    }
    if rs_report:
        metadata["remote_sensing_contract"] = rs_report
    return (repaired, metadata)
