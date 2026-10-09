"""Download the frozen FastEmbed checkpoint into ./model/."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from _common import REPO, sha256, write_json

HUB_DIR = "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"


def load_settings(root: Path):
    from agentgs.config.algorithm import load_algorithm

    return load_algorithm(root)


def resolve_cache(root: Path, _settings) -> Path:
    """The FastEmbed snapshot directory itself, directly under ./model/."""
    return root / "model" / HUB_DIR


def resolve_index(root: Path, settings) -> Path:
    relative = settings.index_path
    if relative.startswith("data/"):
        from agentgs.datasets.layout import prepared_path

        return prepared_path(root, relative.removeprefix("data/"))
    path = Path(relative)
    return path if path.is_absolute() else root / path


def locate_model_directory(cache: Path, model_name: str) -> Path:
    """Find the single ONNX snapshot LocalEmbedding will load."""
    if not cache.is_dir():
        raise FileNotFoundError(f"Frozen embedding cache missing: {cache}")
    needle = model_name.rsplit("/", 1)[-1].casefold()
    candidates = [
        path.parent
        for path in cache.rglob("model_optimized.onnx")
        if needle in str(path).casefold() and path.is_file() and path.stat().st_size
    ]
    if len(candidates) != 1:
        raise ValueError("Frozen embedding model directory is missing or ambiguous")
    return candidates[0]


def require_fastembed():
    try:
        import fastembed
    except ImportError as exc:
        raise RuntimeError(
            "fastembed is not installed; sync the retrieval extra before preparing embeddings"
        ) from exc
    return fastembed


def materialize_external_links(repo: Path) -> None:
    """Copy blobs that Hugging Face stored beside the snapshot into the snapshot tree.

    The runtime cache is the models-- directory under ./model/. It must not depend on a
    sibling hub blobs directory, which would sit next to the other local checkpoints.
    """
    root = repo.resolve()
    for path in list(repo.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if target.is_relative_to(root):
            continue
        if not target.is_file():
            raise FileNotFoundError(f"Embedding symlink target missing: {path}")
        temporary = path.with_name(path.name + ".materialize")
        with target.open("rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        path.unlink()
        os.replace(temporary, path)


def download_model(cache: Path, model_name: str, threads: int) -> Path:
    # Hugging Face writes models--<repo> inside cache_dir, so the parent is ./model/.
    cache.parent.mkdir(parents=True, exist_ok=True)
    require_fastembed().TextEmbedding(
        model_name=model_name,
        cache_dir=str(cache.parent),
        threads=threads,
        providers=["CPUExecutionProvider"],
        local_files_only=False,
    )
    materialize_external_links(cache)
    return locate_model_directory(cache, model_name)


def relative_to(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def cached_files(cache: Path, root: Path) -> dict[str, str]:
    files = [path for path in cache.rglob("*") if path.is_file()]
    return {relative_to(path, root): sha256(path) for path in sorted(files)}


def report(domain: str, status: str, missing: list[str], file_count: int) -> None:
    print(
        json.dumps(
            {"domain": domain, "status": status, "file_count": file_count, "missing": missing},
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPO)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the cached checkpoint and do not download",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    settings = load_settings(output_root)
    cache = resolve_cache(output_root, settings)
    try:
        model_dir = (
            locate_model_directory(cache, settings.embedding_model)
            if args.offline
            else download_model(cache, settings.embedding_model, settings.embedding_threads)
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        write_json(
            output_root / "data/metadata/prepare_embedding_model.json",
            {
                "domain": "embedding_model",
                "status": "partial",
                "model": settings.embedding_model,
                "cache": relative_to(cache, output_root),
                "missing": [str(exc)],
                "files": {},
            },
        )
        report("embedding_model", "partial", [str(exc)], 0)
        raise SystemExit(2) from exc
    files = cached_files(cache, output_root)
    write_json(
        output_root / "data/metadata/prepare_embedding_model.json",
        {
            "domain": "embedding_model",
            "status": "complete",
            "model": settings.embedding_model,
            "cache": relative_to(cache, output_root),
            "model_dir": relative_to(model_dir, output_root),
            "missing": [],
            "files": files,
        },
    )
    report("embedding_model", "complete", [], len(files))


if __name__ == "__main__":
    main()
