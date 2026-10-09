"""Prepared npz vectors are served without loading the ONNX model."""

from agentgs.client.embedding import LocalEmbedding, load_embeddings, save_embeddings


def test_embedding_cache_roundtrip_and_skips_the_model(tmp_path):
    store = tmp_path / "embedding"
    save_embeddings(
        store / "tools.npz", ["('Tool', 'has the functionality that', 'depth')"], [[0.5, -0.25]]
    )
    save_embeddings(store / "questions.npz", ["Where is the Moho?"], [[0.0, 1.0]])
    loaded = load_embeddings(store / "questions.npz")
    embedder = LocalEmbedding(tmp_path / "missing-model", store=store)
    assert loaded["Where is the Moho?"] == [0.0, 1.0]
    assert embedder.get_text_embedding("Where is the Moho?") == [0.0, 1.0]
    assert embedder.get_text_embedding("('Tool', 'has the functionality that', 'depth')") == [
        0.5,
        -0.25,
    ]
    assert embedder.model is None


def test_dynamic_encodings_are_reused_and_query_mode_is_separate(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    calls = []

    def encode(texts, **kwargs):
        calls.extend(texts)
        return iter([np.array([1.0, 2.0]) for _ in texts])

    model = LocalEmbedding(tmp_path)
    model.model = SimpleNamespace(embed=encode, query_embed=encode)
    model.warmup()
    model.warmup()
    for _ in range(3):
        assert model.get_text_embedding("new") == [1.0, 2.0]
        assert model.get_query_embedding("new") == [1.0, 2.0]
    assert calls == ["geoscience", "new", "new"]
    assert model.stats == {"cache_hits": 4, "encoded_texts": 2, "warmups": 1}
