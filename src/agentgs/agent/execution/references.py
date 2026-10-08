"""Provenance helpers for the existing ArtifactRegistry; no independent store."""

import json
import math
from copy import deepcopy
from pathlib import Path

from agentgs.agent.execution import scope as scope

REFERENCE_SLOTS = {
    "RemoteSensingBatchImageMean": "file_list",
    "RemoteSensingTiffAverage": "file_list",
    "RemoteSensingBatchHotspotPercentage": "file_list",
}
SELECTOR = "_rs_source"
DIAGNOSTICS = "_rs_migration_diagnostics"


def fields(value, pointer=""):
    """Exact JSON pointers; preserve list order and repeated/equal values."""
    yield (pointer, value)
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from fields(child, pointer + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from fields(child, pointer + "/" + str(index))


def file_identity(value):
    if not isinstance(value, str):
        return None
    path = Path(value)
    try:
        if not path.is_absolute() or not path.is_file():
            return None
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except (OSError, ValueError):
        return None


def coverage(value):
    if not isinstance(value, dict):
        return {"status": "unknown", "source": "tool_return", "observations": {}}
    observed = {
        k: deepcopy(value[k])
        for k in (
            "full_cohort_coverage",
            "skipped_acquisitions",
            "missing",
            "coverage",
            "input_count",
            "successful_input_count",
        )
        if k in value
    }
    partial = (
        value.get("full_cohort_coverage") is False
        or bool(value.get("skipped_acquisitions"))
        or bool(value.get("missing"))
        or (value.get("coverage") == "partial")
    )
    return {
        "status": "partial"
        if partial
        else "complete"
        if value.get("full_cohort_coverage") is True
        else "unknown",
        "source": "tool_return",
        "observations": observed,
    }


def ingest_production(registry, tool, step, output, execution_info):
    """Called at the existing actual execution boundary, before context resets."""
    if (
        not execution_info
        or not scope.active()
        or registry.request_id != scope.current_request_id()
        or (execution_info.get("call_id") != scope.current_call_id())
        or (not scope.current_call_id())
    ):
        return []
    try:
        value = json.loads(output) if isinstance(output, str) else deepcopy(output)
    except (ValueError, TypeError):
        value = output
    successful = execution_info.get("succeeded") is True
    pointers = list(fields(value))
    identities = {key: identity for key, child in pointers if (identity := file_identity(child))}
    observed_coverage = coverage(value)
    metadata = {
        "request_id": registry.request_id,
        "call_id": execution_info["call_id"],
        "attempt": execution_info.get("attempt"),
        "output_field": "",
        "output_fields": [key for key, _ in pointers],
        "execution_status": "success" if successful else "failed",
        "metadata_source": "actual_dispatch_and_tool_return",
        "coverage": observed_coverage,
        "file_identities": identities,
        "date": None,
        "entity": None,
        "unit": None,
        "scientific_source_correctness": "unverified",
    }
    if successful:
        previous = getattr(registry, "_production_metadata", None)
        registry._production_metadata = {
            "request_id": registry.request_id,
            "call_id": execution_info["call_id"],
        }
        try:
            added = registry._ingest_legacy_tool_output(tool, step, output)
        finally:
            registry._production_metadata = previous
    else:
        added = []
    roots = [r for r in added if r.semantic_type == "StructuredResult"]
    if not roots:
        added.append(
            registry.add(
                value,
                semantic_type="StructuredResult",
                producer=tool,
                producer_step=step,
                metadata=metadata,
            )
        )
    unique = {r.artifact_id: r for r in added}
    for record in unique.values():
        record_fields = list(fields(record.value))
        observed_fields = [key for key, child in pointers if child == record.value]
        record.metadata.update(
            {
                **metadata,
                "output_field": observed_fields[0] if len(observed_fields) == 1 else None,
                "observed_return_fields": observed_fields,
                "output_fields": [key for key, _ in record_fields],
                "file_identities": {
                    key: identity
                    for key, child in record_fields
                    if (identity := file_identity(child))
                },
            }
        )
        record.valid = successful
    return list(unique.values())


class ArtifactReferenceError(ValueError):
    def __init__(self, code, detail, reference=None):

        def portable(value):
            if isinstance(value, float) and (not math.isfinite(value)):
                return {"nonfinite_value": str(value)}
            if isinstance(value, dict):
                return {key: portable(child) for key, child in value.items()}
            if isinstance(value, list):
                return [portable(child) for child in value]
            return deepcopy(value)

        self.diagnostic = {"code": code, "detail": detail, "reference": portable(reference)}
        super().__init__(code + ": " + detail)


def resolve_reference(reference, registry):
    """Resolve one exact current-request ID and JSON pointer, never choose a source."""

    def fail(code, message):
        raise ArtifactReferenceError(code, message, reference)

    if (
        not isinstance(reference, dict)
        or set(reference) != {"artifact_id", "field"}
        or (not isinstance(reference.get("artifact_id"), str))
        or (not reference.get("artifact_id"))
        or (not isinstance(reference.get("field"), str))
    ):
        fail(
            "INVALID_REFERENCE",
            "Exactly one artifact_id and explicit JSON-pointer field are required",
        )
    if not scope.active() or registry is None or registry.request_id != scope.current_request_id():
        fail("REQUEST_MISMATCH", "Registry is absent or outside its request lifetime")
    matches = [r for r in registry.records if r.artifact_id == reference["artifact_id"]]
    if not matches:
        fail("ARTIFACT_NOT_PRODUCED", "No artifact with this ID in the current registry")
    if len(matches) != 1:
        fail("AMBIGUOUS_BINDING", "More than one record has this exact ID")
    record = matches[0]
    meta = record.metadata
    if meta.get("request_id") != scope.current_request_id():
        fail("REQUEST_MISMATCH", "Artifact belongs to another request")
    if not record.valid or meta.get("execution_status") != "success" or (not meta.get("call_id")):
        fail(
            "ARTIFACT_UNAVAILABLE",
            "Artifact is invalid, failed, expired or lacks an observed successful producer",
        )
    if meta.get("coverage", {}).get("status") == "partial":
        fail(
            "PARTIAL_COVERAGE",
            "Producer explicitly reports partial coverage; no substitute is chosen",
        )
    field = reference["field"]
    values = dict(fields(record.value))
    if field not in meta.get("output_fields", []) or field not in values:
        fail(
            "INVALID_FIELD",
            "Field must be an exact registered JSON pointer; no wildcard or inferred field",
        )
    value = values[field]
    for pointer, child in fields(value, field):
        if isinstance(child, float) and (not math.isfinite(child)):
            fail(
                "ARTIFACT_UNAVAILABLE",
                "Non-finite observation remains unavailable, never replaced by zero",
            )
        if isinstance(child, dict) and coverage(child)["status"] == "partial":
            fail("PARTIAL_COVERAGE", "Referenced returned structure reports partial coverage")
        expected = meta.get("file_identities", {}).get(pointer)
        if expected is not None:
            if file_identity(child) != expected:
                fail(
                    "ARTIFACT_EXPIRED",
                    "Referenced file was removed or its observed identity changed",
                )
        elif isinstance(child, str) and Path(child).suffix.lower() in {".tif", ".tiff", ".npy"}:
            fail(
                "ARTIFACT_UNAVAILABLE",
                "Raster path has no reliable file identity from this production call",
            )
    return (
        deepcopy(value),
        {
            "reference": deepcopy(reference),
            "request_id": registry.request_id,
            "producer": record.producer,
            "producer_step": record.producer_step,
            "call_id": meta["call_id"],
            "output_field": meta.get("output_field"),
            "coverage": deepcopy(meta.get("coverage")),
            "scientific_source_correctness": "unverified",
        },
    )


def contains_reference(value):
    return any((isinstance(child, dict) and SELECTOR in child for _, child in fields(value)))


def requires_explicit_source(tool, registry):
    return (
        scope.active()
        and tool in REFERENCE_SLOTS
        and (registry is not None)
        and (registry.request_id == scope.current_request_id())
        and any((r.metadata.get("call_id") for r in registry.records))
    )


def bind_parameters(tool, params, registry):
    """None means untouched legacy path. Failures return repairable diagnostics."""
    if not scope.active() or not (
        contains_reference(params) or requires_explicit_source(tool, registry)
    ):
        return None
    resolved, diagnostics = ([], [])

    def walk(value):
        if isinstance(value, dict) and SELECTOR in value:
            if set(value) != {SELECTOR}:
                raise ArtifactReferenceError(
                    "INVALID_REFERENCE", "Selector must occupy its own value", value
                )
            result, event = resolve_reference(value[SELECTOR], registry)
            resolved.append(event)
            return result
        if isinstance(value, dict):
            return {key: walk(child) for key, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        return deepcopy(value)

    try:
        if tool not in REFERENCE_SLOTS:
            raise ArtifactReferenceError(
                "UNSUPPORTED_REFERENCE_TOOL",
                "This migration supports only the three evidenced file_list entrypoints",
            )
        if requires_explicit_source(tool, registry):
            slot = REFERENCE_SLOTS[tool]
            raw = params.get(slot) if isinstance(params, dict) else None
            sources = raw if isinstance(raw, list) else [raw]
            if not sources or any(
                (not isinstance(s, dict) or set(s) != {SELECTOR} for s in sources)
            ):
                raise ArtifactReferenceError(
                    "EXPLICIT_REFERENCE_REQUIRED",
                    f"{slot} requires explicit artifact ID/field selectors",
                )
        result = walk(params)
        if tool in REFERENCE_SLOTS and REFERENCE_SLOTS[tool] in result:
            values = result[REFERENCE_SLOTS[tool]]
            if (
                not isinstance(values, list)
                or not values
                or (not all((isinstance(v, str) for v in values)))
            ):
                raise ArtifactReferenceError(
                    "FIELD_TYPE_MISMATCH",
                    "file_list must resolve to a nonempty flat list of file paths",
                )
    except ArtifactReferenceError as exc:
        diagnostics.append(exc.diagnostic)
        result = {DIAGNOSTICS: diagnostics}
    return (
        result,
        {
            "type": "artifact_reference_resolution",
            "resolved": resolved,
            "diagnostics": diagnostics,
            "request_id": scope.current_request_id(),
        },
    )


def binding_error_response(tool_input):
    if not scope.active():
        return None
    try:
        value = json.loads(tool_input)
    except (ValueError, TypeError):
        return None
    if isinstance(value, dict) and value.get(DIAGNOSTICS):
        return json.dumps(
            {
                "status": "error",
                "error": "Explicit artifact reference rejected",
                "diagnostics": value[DIAGNOSTICS],
            },
            ensure_ascii=False,
        )
    return None


def argument_contract(input_type, registry, *, tool_name):
    if (
        tool_name not in REFERENCE_SLOTS
        or not scope.active()
        or registry is None
        or (registry.request_id != scope.current_request_id())
    ):
        return input_type
    records = [r for r in registry.records if r.metadata.get("call_id")]
    if not records:
        return input_type
    catalog = [
        {
            "artifact_id": r.artifact_id,
            "producer": r.producer,
            "call_id": r.metadata.get("call_id"),
            "valid": r.valid,
            "semantic_type": r.semantic_type,
            "returned_fields": r.metadata.get("observed_return_fields", []),
            "coverage": r.metadata.get("coverage", {}).get("status", "unknown"),
            "fields": r.metadata.get("output_fields", []),
        }
        for r in records[-12:]
    ]
    return (
        str(input_type)
        + '\nExplicit artifact references: use {"_rs_source":{"artifact_id":"ID","field":"/exact/json/pointer"}} as a parameter value. Select the source yourself. Preserve list order; no automatic source selection, date correction or scientific validation is provided. For file_list of the three scoped reducers, use a selector returning a list or an ordered list of single-path selectors.\n'
        + json.dumps({"registered_artifacts": catalog}, ensure_ascii=False)
    )
