"""Observable boundaries between initial requirements, closure, and completion."""

from .capabilities import concept_words


def expand_requirements(
    graph, selected, query, *, predecessor_expansion, capability_completion, requirements=None
):
    initial = graph.expand(
        selected,
        query,
        expand_predecessors=False,
        complete_missing=False,
        requirements=requirements,
    )
    expanded = graph.expand(
        selected,
        query,
        expand_predecessors=predecessor_expansion,
        complete_missing=False,
        requirements=requirements,
    )
    requested = concept_words(query)
    covered = (
        set().union(*(concept_words(row["description"]) for row in expanded["capabilities"]))
        if expanded["capabilities"]
        else set()
    )
    support_candidates = {}
    for sid in selected:
        for other in graph.edges[sid]["compatible_skills"]:
            if graph.skills.get(other, {}).get("planning", {}).get("bridge_only"):
                continue
            for cid in graph.edges.get(other, {}).get("order", []):
                terms = concept_words(graph.capabilities[cid]["description"]) & (
                    requested - covered
                )
                if terms:
                    support_candidates[f"{other}::{cid}"] = sorted(terms)
    final = (
        graph.expand(
            selected,
            query,
            expand_predecessors=predecessor_expansion,
            complete_missing=True,
            requirements=requirements,
        )
        if capability_completion
        else expanded
    )

    def instance_ids(stage):
        return [row["id"] for row in stage["instances"]]

    initial_ids, expanded_ids, final_ids = map(instance_ids, (initial, expanded, final))
    completed = [item for item in final_ids if item not in expanded_ids]
    return {
        **final,
        "initial_instances": initial_ids,
        "predecessor_added": [item for item in expanded_ids if item not in initial_ids],
        "support_added": completed,
        "missing_before_completion": expanded["unresolved_capabilities"],
        "support_candidates": support_candidates,
        "unresolved_support_candidates": {
            key: value for key, value in support_candidates.items() if key not in final_ids
        },
        "completed_capabilities": list(
            dict.fromkeys(
                [
                    *final["completed_capabilities"],
                    *(row["capability_id"] for row in final["instances"] if row["id"] in completed),
                ]
            )
        ),
    }
