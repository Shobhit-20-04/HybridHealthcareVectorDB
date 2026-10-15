"""
Unit tests for vector search and hybrid retrieval strategies
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
import numpy as np
from src.retrieval.vector_search import VectorSearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine

def test_cosine_distance_computation():
    engine = VectorSearchEngine(metric="cosine")
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    d_ortho = engine.cosine_distance(u, v)
    assert abs(d_ortho - 1.0) < 1e-5, f"Orthogonal vectors must have cosine distance 1.0, got {d_ortho}"

    d_same = engine.cosine_distance(u, u)
    assert abs(d_same - 0.0) < 1e-5, f"Identical vectors must have cosine distance 0.0, got {d_same}"

def test_exact_vs_ivfflat_search():
    engine = VectorSearchEngine(metric="cosine")
    np.random.seed(42)
    corpus_vectors = np.random.randn(50, 384).astype(np.float32)
    corpus_vectors /= np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    corpus_ids = [f"id-{i}" for i in range(50)]

    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)

    # Exact search
    exact_res = engine.search_exact(query, corpus_vectors, corpus_ids, top_k=5)
    assert len(exact_res) == 5

    # IVFFlat search
    ivf_idx = engine.build_ivfflat_index(corpus_vectors, corpus_ids, n_lists=5)
    ivf_res = engine.search_ivfflat(query, ivf_idx, top_k=5, probes=3)
    assert len(ivf_res) <= 5

def test_hybrid_pre_vs_post_filtering():
    hybrid_engine = HybridSearchEngine(metric="cosine")
    np.random.seed(42)

    corpus = []
    for i in range(40):
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        corpus.append({
            "chunk_id": f"chunk-{i}",
            "department": "Cardiology" if i < 10 else "Neurology",
            "embedding": vec.tolist()
        })

    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)

    # Pre-filtering with narrow filter
    pre_res = hybrid_engine.pre_filter_search(query, corpus, {"department": "Cardiology"}, top_k=5)
    assert len(pre_res) == 5
    for r in pre_res:
        assert r["department"] == "Cardiology"

    # Post-filtering with narrow filter
    post_res = hybrid_engine.post_filter_search(query, corpus, {"department": "Cardiology"}, top_k=5, oversample_factor=4)
    assert len(post_res) <= 5
    for r in post_res:
        assert r["department"] == "Cardiology"
