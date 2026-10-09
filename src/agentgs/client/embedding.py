"""Original FastEmbed encodings over a locally prepared, fingerprinted model cache."""

import threading
from pathlib import Path


class LocalEmbedding:
    def __init__(
        self,
        cache: Path,
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        threads=4,
        batch_size=8,
    ):
        self.cache, self.model_name, self.threads, self.batch_size = (
            cache,
            model,
            threads,
            batch_size,
        )
        self.model = None
        self._lock = threading.RLock()

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
            vector = next(iter(self._model().embed([text], batch_size=self.batch_size)))
            return [float(value) for value in vector.tolist()]

    def get_query_embedding(self, text):
        with self._lock:
            vector = next(iter(self._model().query_embed([text])))
            return [float(value) for value in vector.tolist()]
