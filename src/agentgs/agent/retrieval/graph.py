"""Persisted graph traversal and heap-based embedding ranking used by the baseline."""

import heapq


def get_top_k_embeddings(query_embedding, embeddings, similarity_top_k=None, embedding_ids=None):
    import numpy as np

    ids = embedding_ids if embedding_ids is not None else list(range(len(embeddings)))
    query, vectors = np.array(query_embedding), np.array(embeddings)
    heap = []
    for i, vector in enumerate(vectors):
        score = np.dot(query, vector) / (np.linalg.norm(query) * np.linalg.norm(vector))
        heapq.heappush(heap, (score, ids[i]))
        if similarity_top_k and len(heap) > similarity_top_k:
            heapq.heappop(heap)
    rows = sorted(heap, key=lambda item: item[0], reverse=True)
    return [score for score, _ in rows], [name for _, name in rows]


class GraphStore:
    def __init__(self, graph_dict):
        self.graph_dict = graph_dict

    def _get_rel_map(self, subject, depth=2, limit=30):
        if depth == 0:
            return []
        rows, count = [], 0
        for relation, target in self.graph_dict.get(subject, []):
            if count >= limit:
                break
            rows.append([subject, relation, target])
            # Preserve the historical recursive default limit and subject order.
            rows += self._get_rel_map(target, depth - 1)
            count += 1
        return rows

    def get_rel_map(self, subjects=None, depth=2, limit=30):
        subjects = list(self.graph_dict) if subjects is None else subjects
        all_rows = {subject: self._get_rel_map(subject, depth, limit) for subject in subjects}
        result, count = {}, 0
        for subject, rows in all_rows.items():
            if count + len(rows) > limit:
                result[subject] = rows[: limit - count]
                break
            result[subject] = rows
            count += len(rows)
        return result
