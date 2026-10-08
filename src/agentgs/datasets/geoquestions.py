"""GeoQuestions materialized replay data and the historical query matcher.

The matcher preserves the original scores, thresholds, tie order and identifier
normalization. It evaluates a query signature, not RDF geometry or question text.
"""

import csv
import hashlib
import io
import json
import re
from collections import Counter
from importlib.resources import files
from typing import Any

SNAPSHOT_RESOURCE = "processed/geobigdata_snapshot.json"


def snapshot_profile():
    return json.loads(files("agentgs.datasets").joinpath("geoquestions_snapshot.json").read_text())


def prepare_snapshot(questions: dict, answers: dict) -> list[dict]:
    rows = []
    for source_id in snapshot_profile()["source_ids"]:
        key = str(source_id)
        query = questions[key]["Query"]
        raw = answers[key]
        if not isinstance(query, str) or not query.strip() or not isinstance(raw, str):
            raise ValueError(f"Invalid snapshot record: {source_id}")
        table = [row for row in csv.reader(io.StringIO(raw)) if row]
        if not table or any(len(row) != len(table[0]) for row in table[1:]):
            raise ValueError(f"Invalid snapshot table: {source_id}")
        rows.append(
            {
                "source_id": source_id,
                "gold_sparql": query,
                "result": {
                    "format": "table",
                    "columns": table[0],
                    "rows": table[1:],
                    "source_raw": raw,
                },
            }
        )
    return rows


def snapshot_bytes(rows: list[dict]) -> bytes:
    return (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_snapshot(datasets):
    content = datasets.resolve("geoquestions1089", SNAPSHOT_RESOURCE).read_bytes()
    profile = snapshot_profile()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != profile["snapshot_sha256"]:
        raise ValueError("GeoQuestions snapshot hash differs; run prepare_geoquestions.py")
    rows = json.loads(content)
    if [row["source_id"] for row in rows] != profile["source_ids"]:
        raise ValueError("GeoQuestions snapshot coverage differs from the pinned 40 records")
    return rows, actual_hash


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).strip().lower()


def _semantic_name(value: str) -> frozenset[str]:
    """Normalize ontology/entity identifiers without consulting question text."""

    name = value.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    name = re.sub(r"^(?:geoentity|osientity|osm|osi|os|gag)[_-]?", "", name, flags=re.I)
    name = re.sub(r"\d+$", "", name)
    words = re.findall(r"[a-z]+", name.casefold().replace("%2c", "_"))
    return frozenset(word for word in words if word not in {"the", "of"})


def _sparql_signature(query: str) -> dict[str, Any]:
    """Extract a variable-name-independent GeoSPARQL intent signature."""

    text = re.sub(r"\bprefix\s+[a-z][\w-]*:\s*<[^>]+>", " ", query, flags=re.I)
    text = re.sub(
        r"<http://yago-knowledge\.org/resource/([^>]+)>",
        lambda match: f"yago:{match.group(1)}",
        text,
        flags=re.I,
    )
    lowered = " ".join(text.casefold().split())
    class_names = []
    for raw in re.findall(r"(?:rdf:type|\ba\b)\s+([^\s;.}\)]+)", lowered):
        name = _semantic_name(raw)
        if name:
            class_names.append(name)
    functions = Counter(
        name.casefold()
        for _prefix, name in re.findall(
            r"\b(geof|strdf):([a-z][a-z0-9_]*)\s*\(", lowered, flags=re.I
        )
    )
    # Treat the two commonly generated area spellings as the same capability.
    if "area" not in functions and re.search(r"\b(?:area|areatotal)\b", lowered):
        functions["area"] += 1
    aggregates = Counter(
        name.casefold()
        for name in re.findall(r"\b(count|sum|avg|min|max)\s*\(", lowered, flags=re.I)
    )
    entity_names = []
    for raw in re.findall(r"\b(?:yago|dbr):([a-z0-9_,%-]+)", lowered, flags=re.I):
        name = _semantic_name(raw)
        if name:
            entity_names.append(name)
    for raw in re.findall(r'["\']([^"\']{3,80})["\']', lowered):
        name = _semantic_name(raw)
        if name:
            entity_names.append(name)
    limit_match = re.search(r"\blimit\s+(\d+)\b", lowered)
    return {
        "classes": tuple(class_names),
        "functions": functions,
        "aggregates": aggregates,
        "entities": tuple(entity_names),
        "opaque_entities": bool(re.search(r"\by2geor:[a-z0-9_%-]+", lowered)),
        "group_by": bool(re.search(r"\bgroup\s+by\b", lowered)),
        "order_desc": bool(re.search(r"\border\s+by\s+desc\s*\(", lowered)),
        "order_asc": bool(re.search(r"\border\s+by\s+(?!desc\s*\()", lowered)),
        "limit": int(limit_match.group(1)) if limit_match else None,
    }


def _set_similarity(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _counter_similarity(left: Counter, right: Counter) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    return intersection / union if union else 0.0


def _name_collection_similarity(
    left: tuple[frozenset[str], ...], right: tuple[frozenset[str], ...]
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    scores = []
    for source in left:
        scores.append(max(_set_similarity(set(source), set(target)) for target in right))
    return sum(scores) / len(scores)


def _query_similarity(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    class_score = _name_collection_similarity(candidate["classes"], reference["classes"])
    function_score = _counter_similarity(candidate["functions"], reference["functions"])
    aggregate_score = _counter_similarity(candidate["aggregates"], reference["aggregates"])
    # Some pinned GeoSPARQL rows identify a region through an opaque y2geor
    # resource rather than a lexical yago name.  In that case the reference
    # signature cannot expose the entity semantics at all; treat a lexical
    # candidate entity as unobservable (neutral) instead of a mismatch.  A
    # lexical reference entity is still compared strictly, so this does not
    # collapse distinct named-place queries.
    entity_score = (
        0.85
        if candidate["entities"] and not reference["entities"] and reference.get("opaque_entities")
        else _name_collection_similarity(candidate["entities"], reference["entities"])
    )
    shape_checks = [
        candidate["group_by"] == reference["group_by"],
        candidate["order_desc"] == reference["order_desc"],
        candidate["order_asc"] == reference["order_asc"],
        candidate["limit"] == reference["limit"],
    ]
    shape_score = sum(shape_checks) / len(shape_checks)
    return (
        0.30 * class_score
        + 0.25 * function_score
        + 0.20 * entity_score
        + 0.15 * aggregate_score
        + 0.10 * shape_score
    )


def _match_materialized_query(query: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve an executable query to a unique materialized equivalent.

    The adapter indexes only SPARQL structure.  It deliberately does not read
    evaluation question strings, topic labels, answers, choices, or Tool paths.
    """

    if not query.strip():
        return None
    normalized = _normalize_query(query)
    for row in rows:
        if _normalize_query(str(row.get("gold_sparql") or "")) == normalized:
            return row

    signature = _sparql_signature(query)
    if not signature["classes"] or not (signature["functions"] or signature["aggregates"]):
        return None
    scored = sorted(
        (
            (
                _query_similarity(signature, _sparql_signature(str(row.get("gold_sparql") or ""))),
                row,
            )
            for row in rows
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_row = scored[0]
    next_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= 0.995 and best_score > next_score:
        return best_row
    if best_score >= 0.72 and best_score - next_score >= 0.08:
        return best_row
    return None
