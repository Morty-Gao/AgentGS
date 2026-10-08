"""Freeze replay decisions by executing only AST-extracted historical pure functions."""

import argparse
import ast
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from agentgs.datasets.geoquestions import snapshot_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(source.parent):
        parser.error("Use a fresh output outside the historical source directory")
    content = source.read_bytes()
    identity = snapshot_profile()["matching_source"]
    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
        parser.error("Historical matcher source differs from the pinned hash")
    names = {
        "_normalize_query",
        "_semantic_name",
        "_sparql_signature",
        "_set_similarity",
        "_counter_similarity",
        "_name_collection_similarity",
        "_query_similarity",
        "_match_materialized_query",
        "_geosparql",
    }
    tree = ast.parse(content)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        parser.error("Historical matching functions are incomplete")
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    fixture = root / "src/tests/fixtures/geoquestions/snapshot.json"
    raw = fixture.read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot_profile()["snapshot_sha256"]:
        parser.error("Replay data differs from the pinned snapshot")
    rows = json.loads(raw)
    cases = []
    for row in rows:
        sid, query = row["source_id"], row["gold_sparql"]
        for kind, params in [
            ("source-id", {"source_id": sid}),
            ("exact", {"query": query}),
            ("renamed", {"query": re.sub(r"\?([A-Za-z_][\w]*)", r"?renamed_\1", query)}),
        ]:
            cases.append({"id": f"{kind}-{sid}", "params": params})
    cases.extend(
        [
            {
                "id": "conflicting-query",
                "params": {"source_id": rows[0]["source_id"], "query": rows[1]["gold_sparql"]},
            },
            {
                "id": "missing-id-query-fallback",
                "params": {"source_id": 999999, "query": rows[1]["gold_sparql"]},
            },
            {"id": "query-alias", "params": {"sparql": rows[0]["gold_sparql"]}},
            {"id": "unknown-id", "params": {"source_id": 999999}},
            {"id": "unknown-query", "params": {"query": "ASK { <urn:unknown> a <urn:unknown> }"}},
        ]
    )
    with tempfile.TemporaryDirectory(prefix="agentgs-snapshot-oracle-") as temp:
        dataset = Path(temp)
        (dataset / "2.机器学习和大数据分析.json").write_bytes(raw)
        namespace = {
            "re": re,
            "json": json,
            "Counter": Counter,
            "Any": Any,
            "_DATASET_ROOT": dataset,
            "_result": lambda **payload: json.dumps(payload),
        }
        # No historical runtime imports, tools, endpoint calls or model execution.
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        for case in cases:
            try:
                case["expected"] = json.loads(namespace["_geosparql"](case["params"]))
            except ValueError:
                case["expected_error"] = "ValueError"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "oracle": "AST-extracted original _geosparql and matcher; offline replay only",
                "source_sha256": identity["sha256"],
                "snapshot_sha256": snapshot_profile()["snapshot_sha256"],
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "cases": len(cases)}))


if __name__ == "__main__":
    main()
