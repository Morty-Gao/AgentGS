"""Predeclared information-access strata, independent of experimental outcomes."""


def access_class(example):
    if example.category == "GeoBigDataTools":
        return "materialized_query_lookup"
    return "shared_data_candidate"


def select_scope(examples, scope, *, planning_mode):
    if scope not in {"full", "common_data"}:
        raise ValueError("benchmark_scope must be full or common_data")
    excluded = [e for e in examples if access_class(e) == "materialized_query_lookup"]
    selected = examples if scope == "full" else [e for e in examples if e not in excluded]
    return selected, {
        "scope": scope,
        "excluded": [
            {
                "question_id": e.question_id,
                "reason": "materialized_query_lookup_unavailable_to_bash",
            }
            for e in excluded
        ]
        if scope == "common_data"
        else [],
        "strata": {str(e.question_id): access_class(e) for e in selected},
        "guarantee": "known_snapshot_asymmetry_excluded; resource_presence_and_task_solvability_require_preflight",
    }
