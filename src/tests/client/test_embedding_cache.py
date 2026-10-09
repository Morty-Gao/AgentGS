"""Prepared npz vectors are served without loading the ONNX model."""

from agentgs.client.embedding import LocalEmbedding, load_embeddings, save_embeddings


def test_embedding_cache_roundtrip_and_skips_the_model(tmp_path):
    store = tmp_path / "embedding"
    save_embeddings(store / "tools.npz", ["('Tool', 'has the functionality that', 'depth')"], [[0.5, -0.25]])
    save_embeddings(store / "questions.npz", ["Where is the Moho?"], [[0.0, 1.0]])
    loaded = load_embeddings(store / "questions.npz")
    embedder = LocalEmbedding(tmp_path / "missing-model", store=store)
    assert loaded["Where is the Moho?"] == [0.0, 1.0]
    assert embedder.get_text_embedding("Where is the Moho?") == [0.0, 1.0]
    assert embedder.get_query_embedding("('Tool', 'has the functionality that', 'depth')") == [
        0.5,
        -0.25,
    ]
    assert embedder.model is None
