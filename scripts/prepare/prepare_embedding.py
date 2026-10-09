"""Encode tool relations and GeoToolEval questions into ./embedding/*.npz.

Retrieval reads these vectors instead of calling the embedding model. The
relation key is repr((subject, relation, object)). The question key is the
benchmark question text passed to the retriever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import REPO, sha256, write_json
from prepare_embedding_model import (
    load_settings,
    locate_model_directory,
    relative_to,
    report,
    require_fastembed,
    resolve_cache,
    resolve_index,
)

RELATION = "has the functionality that"


def resolve_store(root: Path) -> Path:
    return root / "embedding"


def functionality_keys(graph: dict) -> list[str]:
    keys = [
        repr((subject, relation, target))
        for subject, edges in graph.items()
        for relation, target in edges
        if relation == RELATION
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate functionality relations in the tool graph")
    if not keys:
        raise ValueError("Tool graph has no functionality relations")
    return keys


def load_graph(index_dir: Path) -> dict:
    graph_path = index_dir / "graph_store.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Frozen tool graph is missing: {graph_path}")
    return json.loads(graph_path.read_text(encoding="utf-8"))["graph_dict"]


def question_texts(root: Path) -> list[str]:
    from agentgs.datasets.geotooleval import benchmark_dataset
    from agentgs.evaluation.datasets.geotooleval import load_tasks

    texts = []
    seen = set()
    for task in load_tasks(benchmark_dataset(root)):
        text = task.question
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    if not texts:
        raise ValueError("GeoToolEval has no questions to embed")
    return texts


def encode(embedder, texts: list[str], label: str) -> list[list[float]]:
    vectors = []
    total = len(texts)
    for done, text in enumerate(texts, start=1):
        vectors.append(embedder.get_text_embedding(text))
        if done == total or done % 20 == 0:
            print(f"embedded {label} {done}/{total}", file=sys.stderr, flush=True)
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("Inconsistent embedding dimension")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    settings = load_settings(output_root)
    model_cache = resolve_cache(output_root, settings)
    index_dir = resolve_index(output_root, settings)
    store = resolve_store(output_root)
    manifest_path = output_root / "data/metadata/prepare_embedding.json"
    try:
        locate_model_directory(model_cache, settings.embedding_model)
        require_fastembed()
        relations = functionality_keys(load_graph(index_dir))
        questions = question_texts(output_root)
        from agentgs.client.embedding import LocalEmbedding, save_embeddings

        embedder = LocalEmbedding(
            model_cache,
            settings.embedding_model,
            settings.embedding_threads,
            settings.embedding_batch_size,
        )
        relation_vectors = encode(embedder, relations, "tools")
        question_vectors = encode(embedder, questions, "questions")
        save_embeddings(store / "tools.npz", relations, relation_vectors)
        save_embeddings(store / "questions.npz", questions, question_vectors)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        write_json(
            manifest_path,
            {
                "domain": "embedding",
                "status": "partial",
                "model": settings.embedding_model,
                "cache": relative_to(model_cache, output_root),
                "store": relative_to(store, output_root),
                "missing": [str(exc)],
                "files": {},
            },
        )
        report("embedding", "partial", [str(exc)], 0)
        raise SystemExit(2) from exc
    files = [store / "tools.npz", store / "questions.npz"]
    write_json(
        manifest_path,
        {
            "domain": "embedding",
            "status": "complete",
            "model": settings.embedding_model,
            "cache": relative_to(model_cache, output_root),
            "store": relative_to(store, output_root),
            "relations": len(relations),
            "questions": len(questions),
            "dimension": len(relation_vectors[0]),
            "missing": [],
            "files": {relative_to(path, output_root): sha256(path) for path in files},
        },
    )
    report("embedding", "complete", [], len(files))


if __name__ == "__main__":
    main()
