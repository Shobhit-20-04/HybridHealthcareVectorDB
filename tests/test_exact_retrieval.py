"""
Unit & Integration Tests for Exact Semantic Retrieval Baseline
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Verifies:
1. Ground-truth ranking order (strictly ascending cosine distance).
2. Mathematical relationship between cosine distance and cosine similarity (S_C = 1 - D_C).
3. Input validation: empty queries and top-k boundary conditions.
4. Error handling for missing embeddings and model dimension mismatches.
5. Deterministic retrieval reproducibility across multiple executions.
6. Execution plan verification (guaranteed sequential scan without approximate indexes).
"""

import pytest
import math
from src.retrieval.exact_search import (
    ExactSemanticSearchEngine,
    RetrievalError,
    NoEmbeddingsFoundError
)
from src.embeddings.text_embedder import DimensionMismatchError

@pytest.fixture(scope="module")
def search_engine():
    """Initializes standard search engine fixture using 384-d all-MiniLM-L6-v2."""
    engine = ExactSemanticSearchEngine(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension=384
    )
    return engine

def test_exact_ranking_order_and_metrics(search_engine):
    """Verify that results are strictly ordered by ascending cosine distance."""
    response = search_engine.search("fever, purulent sputum, and pleuritic chest pain", top_k=5)
    results = response["results"]

    assert len(results) > 0
    assert response["returned_count"] == len(results)

    # 1. Check ranks are contiguous 1..N
    expected_ranks = list(range(1, len(results) + 1))
    actual_ranks = [r["rank"] for r in results]
    assert actual_ranks == expected_ranks

    # 2. Check distances are strictly sorted ascending
    distances = [r["cosine_distance"] for r in results]
    assert distances == sorted(distances)

    # 3. Check mathematical invariant: Cosine Similarity = 1.0 - Cosine Distance
    for r in results:
        dist = r["cosine_distance"]
        sim = r["cosine_similarity"]
        assert 0.0 <= dist <= 2.0
        assert -1.0 <= sim <= 1.0
        assert math.isclose(sim, 1.0 - dist, abs_tol=1e-5)

def test_empty_and_whitespace_query_validation(search_engine):
    """Verify that empty or whitespace-only queries raise ValueError."""
    for empty_input in ["", "   ", "\t\t", "\n\n"]:
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            search_engine.search(empty_input, top_k=5)

def test_top_k_boundary_validation(search_engine):
    """Verify top-k parameter validation and limits."""
    # Invalid k <= 0
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_engine.search("chest discomfort", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_engine.search("chest discomfort", top_k=-3)

    # Valid k=1
    res1 = search_engine.search("chest discomfort", top_k=1)
    assert len(res1["results"]) == 1
    assert res1["results"][0]["rank"] == 1

    # Valid k=3
    res3 = search_engine.search("chest discomfort", top_k=3)
    assert len(res3["results"]) == 3
    assert [r["rank"] for r in res3["results"]] == [1, 2, 3]

def test_missing_embeddings_error_handling():
    """Verify error handling when querying for an unregistered or unpopulated model."""
    unregistered_engine = ExactSemanticSearchEngine(
        model_id="unregistered/fake-model-999",
        expected_dimension=384
    )
    with pytest.raises(RetrievalError):
        unregistered_engine.search("cardiac arrest", top_k=5)

def test_model_dimension_mismatch_check():
    """Verify that dimension mismatches between engine and database column raise error."""
    mismatched_engine = ExactSemanticSearchEngine(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension=768  # Schema expects 384
    )
    with pytest.raises(DimensionMismatchError):
        mismatched_engine.search("cardiac arrest", top_k=5)

def test_retrieval_reproducibility(search_engine):
    """Verify that executing the exact same query multiple times yields bit-identical results."""
    query = "elevated troponin and acute myocardial infarction"
    run1 = search_engine.search(query, top_k=5)
    run2 = search_engine.search(query, top_k=5)

    assert run1["returned_count"] == run2["returned_count"]
    for r1, r2 in zip(run1["results"], run2["results"]):
        assert r1["chunk_id"] == r2["chunk_id"]
        assert r1["rank"] == r2["rank"]
        assert math.isclose(r1["cosine_distance"], r2["cosine_distance"], abs_tol=1e-7)
        assert math.isclose(r1["cosine_similarity"], r2["cosine_similarity"], abs_tol=1e-7)

def test_exact_sequential_scan_execution_plan(search_engine):
    """Verify that EXPLAIN ANALYZE confirms a pure Seq Scan without index traversal."""
    response = search_engine.search("shortness of breath", top_k=3, explain=True)
    plan_text = "\n".join(response["query_plan"])

    assert "Seq Scan on chunk_embeddings" in plan_text
    assert "Index Scan" not in plan_text
    assert "Bitmap Index Scan" not in plan_text
