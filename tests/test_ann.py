# Optional ANN test: compares hnswlib k-NN results to brute-force cosine k-NN.
# Skips automatically if hnswlib is not installed.

from __future__ import annotations

import numpy as np
import pytest

hnswlib = pytest.importorskip("hnswlib", reason="hnswlib not installed")


def _cosine_topk_bruteforce(X: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """Return indices of brute-force top-k neighbors by cosine similarity."""
    # Normalize to unit vectors
    def _norm(A: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        return A / n

    Xn = _norm(X)
    qn = q / (np.linalg.norm(q) + 1e-12)
    sims = Xn @ qn  # (N,)
    return np.argsort(-sims)[:k]


def test_hnsw_recall_vs_bruteforce():
    """
    Build a small clustered dataset, index with hnswlib (cosine) and
    measure recall@k against brute force. Expect high recall.
    """
    rng = np.random.default_rng(42)
    N = 500          # database size
    D = 32           # vector dim
    C = 5            # clusters
    K = 10           # top-k
    Q = 30           # #queries

    # Create clustered unit vectors for an easy/nearly deterministic task
    centers = rng.normal(size=(C, D)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    X = []
    for i in range(N):
        c = i % C
        v = centers[c] + 0.01 * rng.normal(size=D).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        X.append(v)
    X = np.stack(X, axis=0).astype(np.float32)

    # Build HNSW index (cosine)
    index = hnswlib.Index(space="cosine", dim=D)
    index.init_index(max_elements=N, ef_construction=200, M=16)
    index.add_items(X, np.arange(N))
    index.set_ef(64)  # higher ef -> better recall

    # Evaluate recall@K on held-out queries (sample from X)
    q_idx = rng.choice(N, size=Q, replace=False)
    recall_hits = 0

    for qi in q_idx:
        q = X[qi]
        # ANN
        labels, _ = index.knn_query(q, k=K)
        ann_topk = labels[0]
        # Brute force oracle
        bf_topk = _cosine_topk_bruteforce(X, q, k=K)
        # count intersection size
        recall_hits += len(set(ann_topk.tolist()) & set(bf_topk.tolist()))

    recall_at_k = recall_hits / (Q * K)
    # Expect strong recall on this easy dataset
    assert recall_at_k >= 0.95, f"recall@{K} too low: {recall_at_k:.3f}"