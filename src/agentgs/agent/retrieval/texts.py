"""Shared, exact encoder inputs for tool retrieval and offline preparation."""

from .graph import GraphStore


def relation_text(row):
    return "('" + row[0] + "', '" + row[1] + "', '" + row[2] + "')"


def anchor_text(relation):
    return "There is an existing tool:" + relation.replace("'", "").replace("(", "").replace(
        ")", ""
    )


def context_texts(graph, depth=3, limit=1000):
    store, texts = GraphStore(graph), set()
    for name, edges in graph.items():
        if not any(predicate == "has the functionality that" for predicate, _ in edges):
            continue
        rows = list(
            dict.fromkeys(
                relation_text(row)
                for row in store.get_rel_map([name], depth, limit).get(name, [])
                if "functionality" in relation_text(row)
            )
        )
        if rows:
            texts.update(row + anchor_text(rows[0]) for row in rows[1:])
    return sorted(texts)
