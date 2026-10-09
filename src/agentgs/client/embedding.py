"""Original FastEmbed encodings over a locally prepared, fingerprinted model cache.

Prepared benchmark vectors live in ``./embedding/*.npz``. A cached text is returned
without loading the ONNX model.
"""

import threading
from pathlib import Path

import numpy as np

CACHE_FILES = ("tools.npz", "questions.npz")


def save_embeddings(path: Path, texts: list[str], vectors: list[list[float]]) -> None:
    if len(texts) != len(vectors):
        raise ValueError("Embedding texts and vectors have different lengths")
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(texts) or not len(texts):
        raise ValueError("Embedding cache must be a non-empty matrix")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez(stream, text=np.asarray(texts), vector=array)
    temporary.replace(path)


def load_embeddings(path: Path) -> dict[str, list[float]]:
    with np.load(path, allow_pickle=False) as store:
        texts = store["text"]
        vectors = store["vector"]
    if len(texts) != len(vectors):
        raise ValueError(f"Embedding cache is misaligned: {path}")
    return {
        str(text): [float(value) for value in row]
        for text, row in zip(texts, vectors, strict=True)
    }


class LocalEmbedding:
    def __init__(
        self,
        cache: Path,
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        threads=4,
        batch_size=8,
        store: Path | None = None,
    ):
        self.cache, self.model_name, self.threads, self.batch_size = (
            cache,
            model,
            threads,
            batch_size,
        )
        self.model = None
        self._lock = threading.RLock()
        self._stored: dict[str, list[float]] = {}
        if store is not None:
            for name in CACHE_FILES:
                path = store / name
                if path.is_file():
                    self._stored.update(load_embeddings(path))

    def _model(self):
        with self._lock:
            return self._load_model()

    def _load_model(self):
        if self.model is None:
            if not self.cache.is_dir():
                raise FileNotFoundError(f"Frozen embedding cache missing: {self.cache}")
            from fastembed import TextEmbedding

            candidates = [
                path.parent
                for path in self.cache.rglob("model_optimized.onnx")
                if self.model_name.rsplit("/", 1)[-1].casefold() in str(path).casefold()
            ]
            if len(candidates) != 1:
                raise ValueError("Frozen embedding model directory is missing or ambiguous")
            self.model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache),
                threads=self.threads,
                local_files_only=True,
                specific_model_path=str(candidates[0]),
            )
        return self.model

    def get_text_embedding(self, text):
        with self._lock:
            cached = self._stored.get(text)
            if cached is not None:
                return list(cached)
            vector = next(iter(self._model().embed([text], batch_size=self.batch_size)))
            return [float(value) for value in vector.tolist()]

    def get_query_embedding(self, text):
        with self._lock:
            cached = self._stored.get(text)
            if cached is not None:
                return list(cached)
            vector = next(iter(self._model().query_embed([text])))
            return [float(value) for value in vector.tolist()]
