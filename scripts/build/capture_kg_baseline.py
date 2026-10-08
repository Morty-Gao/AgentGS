"""Freeze retrieval decisions from original KG code and its installed SDK source."""

import argparse
import ast
import asyncio
import contextlib
import hashlib
import heapq
import io
import json
import logging
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import numpy as np


def evaluate(nodes, namespace, filename):
    header = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[header, *nodes], type_ignores=[])),
            str(filename),
            "exec",
        ),
        namespace,
    )


def capture(source):
    sdk = source / ".venv/Lib/site-packages/llama_index/core"
    files = [
        source / "scripts/custom_kg_retrievers.py",
        source / "app/llms/planning/tool_routing.py",
        source / "app/tools/tool_input_contracts.py",
        sdk / "base/embeddings/base.py",
        sdk / "indices/query/embedding_utils.py",
        sdk / "graph_stores/simple.py",
    ]
    identities = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    namespace = {
        "np": np,
        "heapq": heapq,
        "Enum": Enum,
        "re": __import__("re"),
        "json": json,
        "Path": Path,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "logger": logging.getLogger("source_kg"),
    }
    binding = ast.parse(files[2].read_text())
    evaluate(
        [
            n
            for n in binding.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "split_question_resource"
            or isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "RESOURCE_MARKER" for t in n.targets)
        ],
        namespace,
        files[2],
    )
    routing = ast.parse(files[1].read_text())
    evaluate(
        [n for n in routing.body if not isinstance(n, (ast.Import, ast.ImportFrom))],
        namespace,
        files[1],
    )
    for name in ("app", "app.llms", "app.llms.planning"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    module = types.ModuleType("app.llms.planning.tool_routing")
    module.infer_tools = namespace["infer_tools"]
    sys.modules[module.__name__] = module
    tree = ast.parse(files[3].read_text())
    evaluate(
        [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "SimilarityMode"
            or isinstance(n, ast.FunctionDef)
            and n.name == "similarity"
        ],
        namespace,
        files[3],
    )
    namespace["default_similarity_fn"] = namespace["similarity"]
    tree = ast.parse(files[4].read_text())
    evaluate(
        [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_top_k_embeddings"
        ],
        namespace,
        files[4],
    )
    graph = next(
        n
        for n in ast.parse(files[5].read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "SimpleGraphStoreData"
    )
    graph.decorator_list = []
    graph.bases = []
    graph.body = [n for n in graph.body if isinstance(n, ast.FunctionDef)]
    evaluate([graph], namespace, files[5])
    cls = next(
        n
        for n in ast.parse(files[0].read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "KGTableRetriever"
    )
    cls.bases, cls.decorator_list = [], []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name != "__init__"
    ]
    evaluate([cls], namespace, files[0])
    names = [
        "RemoteSensingBatchImageMean",
        "RemoteSensingMean",
        "RemoteSensingDifference",
        "RemoteSensingDivision",
        "DeepEarthQueryUSGSCatalog",
        "NaturalHazardLoadEventTable",
        "NaturalHazardCountRecords",
    ]
    graph_dict = {
        name: [
            ["has the functionality that", f"Calculate {name} statistic"],
            ["inputs", "JSON object"],
            ["outputs", "Numeric result"],
        ]
        for name in names
    }
    vectors = {
        str((name, "has the functionality that", f"Calculate {name} statistic")): [i + 1, 8 - i]
        for i, name in enumerate(names)
    }

    class Embed:
        def __init__(self, dimension):
            self.dimension = dimension

        def get_text_embedding(self, text):
            return [1.0, 2.0] if self.dimension == 2 else [1.0, 2.0, 3.0]

    cases = []
    for next_edges in (False, True):
        graph = json.loads(json.dumps(graph_dict))
        if next_edges:
            for first, second in zip(names, names[1:]):
                graph[first].append(["recommended next tool", second])
        for dimension in (2, 3):
            for query in (
                "Compute the requested statistic.",
                "Count USGS earthquake events in 2011.",
                "Calculate remote sensing raster means.",
                "Count natural disaster event table records.",
            ):
                reference = namespace["KGTableRetriever"]()
                store = namespace["SimpleGraphStoreData"]()
                store.graph_dict = graph
                reference._index_struct = types.SimpleNamespace(embedding_dict=vectors)
                reference._graph_store, reference._embed_model = store, Embed(dimension)
                (
                    reference.similarity_top_k,
                    reference.graph_store_query_depth,
                    reference.max_tools,
                    reference.max_knowledge_sequence,
                ) = 5, 3, 20, 1000
                with contextlib.redirect_stdout(io.StringIO()):
                    result = asyncio.run(
                        reference._retrieve(types.SimpleNamespace(query_str=query))
                    )
                cases.append(
                    {
                        "query": query,
                        "dimension": dimension,
                        "graph": graph,
                        "vectors": vectors,
                        "expected": result,
                    }
                )
    return {"source_files": identities, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing oracles; select a fresh output path")
    result = capture(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
