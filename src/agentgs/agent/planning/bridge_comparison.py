"""Isolated, read-only numerical direct-path comparison for the legacy runtime."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from numbers import Real
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agentgs.agent.execution.artifacts import ArtifactRegistry
from agentgs.agent.execution.feedback import run_tool_with_feedback
from agentgs.agent.planning.bridge import _project

SAFE_OPERANDS = {
    "RemoteSensingMean": {"x"},
    "RemoteSensingDifference": {"a"},
    "RemoteSensingDivision": {"a", "a_values"},
}


class NotComparable(ValueError):
    pass


def _finite_numeric(value):
    if isinstance(value, Real) and (not isinstance(value, bool)):
        return math.isfinite(value)
    return (
        isinstance(value, list) and bool(value) and all((_finite_numeric(item) for item in value))
    )


def _safe_parameters(name, params):
    if name == "RemoteSensingMean":
        allowed = {"x"}
    elif name == "RemoteSensingDifference":
        allowed = {"a", "b"}
    elif name == "RemoteSensingDivision":
        allowed = {"a_values", "b_values"} if "a_values" in params else {"a", "b"}
    else:
        raise NotComparable("endpoint_tool_has_no_read_only_safety_declaration")
    if set(params) != allowed or not all((_finite_numeric(value) for value in params.values())):
        raise NotComparable("secondary_call_requires_only_finite_numeric_parameters")


async def run_shared_bridge_comparisons(state, *, tool_runner, file_path_list=(), trace_list=None):
    """Reuse the successful prefix and run only the direct numerical endpoint once.

    Each callback receives only deep-copied numeric parameters and file context.
    No main artifact registry, output list, result list, or mutable Plan is shared.
    Secondary tool_call events remain nested in the comparison's branch trace.
    """
    if not state:
        return []
    existing = {tuple(row["capability_path"]) for row in state["comparisons"]}
    for route in state["routes"]:
        path = route["capability_path"]
        if not route.get("direct_compatible") or tuple(path) in existing:
            continue
        existing.add(tuple(path))
        record = {
            "capability_path": list(path),
            "skill_id": route["skill_id"],
            "parent_run_id": state["run_id"],
            "run_id": f"bridge-direct-{uuid4().hex}",
            "status": "incomparable",
            "main_path": "declared_bridge",
            "direct_invocation_count": 0,
            "branch_trace": [],
            "scientific_equivalence": "not_established",
        }
        try:
            indices = [state["capability_indices"][cid] for cid in path]
            rows = [state["execution_records"].get(index) for index in indices]
            if not all((row and row["succeeded"] for row in rows)):
                raise NotComparable("declared_bridge_did_not_succeed")
            if indices != sorted(indices) or len(set(indices)) != len(indices):
                raise NotComparable("bridge_execution_order_is_ambiguous")
            first, last = (rows[0], rows[-1])
            name = last["tool_name"]
            if name not in SAFE_OPERANDS:
                raise NotComparable("endpoint_tool_has_no_read_only_safety_declaration")
            declared = json.loads(last["input"])
            direct = deepcopy(declared)
            bindings = route["execution_contract"][path[-1]]["input_bindings"]
            mutable = {
                field: binding
                for field, binding in bindings.items()
                if binding["source"] == path[-2]
            }
            if not mutable or not set(mutable) <= SAFE_OPERANDS[name]:
                raise NotComparable("endpoint_has_no_safe_declared_data_operand")
            if any((binding["source"] in path[1:-2] for binding in bindings.values())):
                raise NotComparable("endpoint_has_bridge_derived_side_input")
            for field, binding in mutable.items():
                projection = deepcopy(binding)
                projection["path"] = binding.get("direct_path", binding["path"])
                direct[field] = _project(first["output"], projection)
            _safe_parameters(name, declared)
            _safe_parameters(name, direct)
            if any(
                (declared[field] != direct[field] for field in declared if field not in mutable)
            ):
                raise NotComparable("non_data_operand_changed")
            branch_registry = deepcopy(
                state["prefix_artifacts"].get(indices[0], ArtifactRegistry())
            )
            branch_file_paths = deepcopy(list(file_path_list))
            branch_trace = record["branch_trace"]
            record.update(
                tool_name=name,
                declared_input=deepcopy(declared),
                direct_input=deepcopy(direct),
                reused_source_execution_index=indices[0],
                fixed_input_fields=sorted(set(declared) - set(mutable)),
                source_output=deepcopy(first["output"]),
                declared_output=deepcopy(last["output"]),
            )
            record["direct_invocation_count"] = 1

            async def fixed_input(_attempt, _history):
                return json.dumps(direct, ensure_ascii=False, allow_nan=False)

            result = await run_tool_with_feedback(
                tool_name=name,
                file_path_list=branch_file_paths,
                input_provider=fixed_input,
                tool_runner=tool_runner,
                max_attempts=1,
                trace_list=branch_trace,
            )
            record["direct_output"] = deepcopy(result.output)
            if not result.succeeded:
                record.update(status="failed", reason="direct_endpoint_did_not_succeed")
            else:
                branch_registry.ingest_tool_output(
                    name, f"{record['run_id']}:endpoint", result.output
                )
                measured = compare_endpoint_results(SimpleNamespace(output=last["output"]), result)
                record.update(measured)
                record["artifact_registry"] = branch_registry.compact_context()
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            record.update(status="incomparable", reason=str(exc))
        except Exception as exc:
            record.update(status="failed", reason=f"comparison_error:{type(exc).__name__}")
        state["comparisons"].append(record)
        if trace_list is not None:
            trace_list.append(
                {"type": "shared_bridge_path_comparison", "content": deepcopy(record)}
            )
    return deepcopy(state["comparisons"])


def compare_endpoint_results(
    declared: Any, direct: Any, *, rtol: float = 1e-09, atol: float = 1e-12
) -> dict[str, Any]:
    """Compare finite numeric payloads, never filenames or scientific validity."""

    def payload(result):
        value = getattr(result, "data", None)
        if value is None:
            try:
                value = json.loads(result.output)
            except (TypeError, ValueError):
                raise NotComparable("endpoint_has_no_structured_numeric_output") from None
        return value

    differences = []
    numeric_count = 0

    def compare(left, right, path="$"):
        nonlocal numeric_count
        if isinstance(left, Real) and (not isinstance(left, bool)):
            if not isinstance(right, Real) or isinstance(right, bool):
                raise NotComparable("endpoint_structure_mismatch")
            if not math.isfinite(left) or not math.isfinite(right):
                raise NotComparable("non_finite_endpoint_output")
            numeric_count += 1
            if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
                differences.append(
                    {
                        "field": path,
                        "declared": left,
                        "direct": right,
                        "absolute_difference": abs(left - right),
                    }
                )
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                raise NotComparable("endpoint_structure_mismatch")
            for key in left:
                label = str(key).casefold()
                if label in {
                    "path",
                    "paths",
                    "file",
                    "files",
                    "filename",
                    "filenames",
                    "uri",
                    "url",
                } or label.endswith(
                    ("_path", "_paths", "_file", "_files", "_filename", "_url", "_uri")
                ):
                    raise NotComparable("artifact_outputs_require_content_comparator")
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise NotComparable("endpoint_structure_mismatch")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{index}]")
        elif isinstance(left, (str, bool)) or left is None:
            if isinstance(left, str) and (
                left.startswith(("/", "http://", "https://", "file://"))
                or "\\" in left
                or left.casefold().endswith(
                    (".tif", ".tiff", ".npy", ".csv", ".nc", ".png", ".json")
                )
            ):
                raise NotComparable("artifact_outputs_require_content_comparator")
            if type(left) is not type(right) or left != right:
                raise NotComparable("endpoint_metadata_or_units_differ")
        else:
            raise NotComparable("unsupported_endpoint_output")

    try:
        compare(payload(declared), payload(direct))
        if not numeric_count:
            raise NotComparable("endpoint_has_no_numeric_measurements")
    except NotComparable as exc:
        return {
            "status": "incomparable",
            "reason": str(exc),
            "scientific_equivalence": "not_established",
        }
    return {
        "status": "different" if differences else "equal",
        "compared_numeric_values": numeric_count,
        "differences": differences,
        "rtol": rtol,
        "atol": atol,
        "scientific_equivalence": "not_established",
        "interpretation": "Observed endpoint values only; scientific equivalence is not established.",
    }
