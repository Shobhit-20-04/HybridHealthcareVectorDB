"""
Automated Test Suite for Controlled Vector Index Experiments
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Validates:
  - PostgreSQL & pgvector system catalog discovery
  - IVFFlat data sufficiency guards
  - HNSW & IVFFlat creation, timing, sizing, and catalog registration
  - Exact Sequential Scan vs HNSW vs IVFFlat plan verification via EXPLAIN (ANALYZE, BUFFERS)
  - Mathematical correctness of Distance 1-Recall@K and Strict ID Recall@K
  - End-to-end controlled benchmark execution and metrics population
"""

import pytest
from typing import List, Tuple
from database.connection import get_connection
from src.benchmark.index_experiment import (
    VectorIndexExperimentRunner,
    InsufficientDataForIndexError
)

@pytest.fixture(scope="module")
def runner():
    """Module-level fixture providing initialized experiment runner."""
    r = VectorIndexExperimentRunner()
    yield r
    # Guarantee standard HNSW index restored after test module completes
    r.restore_default_index()

def test_system_metadata_retrieval(runner):
    """Verifies retrieval of PostgreSQL version, pgvector version, and catalog stats."""
    meta = runner.get_system_metadata()
    assert "postgres_version" in meta
    assert "pgvector_version" in meta
    assert meta["pgvector_version"] == "0.8.6"
    assert meta["model_vectors_count"] > 0
    assert meta["table_disk_size_bytes"] > 0

def test_ivfflat_insufficient_data_guard(runner):
    """Verifies that attempting to build IVFFlat with lists > vector count raises InsufficientDataForIndexError."""
    with pytest.raises(InsufficientDataForIndexError) as exc_info:
        runner.create_ivfflat_index(lists=9999999)
    assert "Insufficient vectors" in str(exc_info.value)

def test_calculate_recalls_mathematics():
    """Tests mathematical edge cases of both distance 1-recall and strict ID recall."""
    # Case 1: Identical retrieved and ground truth (100% both)
    gt = [("id_1", 0.1), ("id_2", 0.2), ("id_3", 0.3)]
    ret = [("id_1", 0.1), ("id_2", 0.2), ("id_3", 0.3)]
    dist_rec, id_rec = VectorIndexExperimentRunner.calculate_recalls(ret, gt, k=3)
    assert dist_rec == 1.0
    assert id_rec == 1.0

    # Case 2: Disjoint IDs with strictly higher distances (0% both)
    ret_worse = [("id_x", 0.5), ("id_y", 0.6), ("id_z", 0.7)]
    dist_rec, id_rec = VectorIndexExperimentRunner.calculate_recalls(ret_worse, gt, k=3)
    assert dist_rec == 0.0
    assert id_rec == 0.0

    # Case 3: Disjoint IDs but tied optimal distances (e.g. repeated synthetic text chunks)
    # Ground truth top-3 distance threshold D_3 = 0.3
    # Retrieved items all have distance 0.2 (better than D_3) but different IDs
    ret_tied = [("id_a", 0.2), ("id_b", 0.25), ("id_c", 0.3)]
    dist_rec, id_rec = VectorIndexExperimentRunner.calculate_recalls(ret_tied, gt, k=3)
    assert dist_rec == 1.0
    assert id_rec == 0.0

    # Case 4: Invalid k <= 0
    with pytest.raises(ValueError):
        VectorIndexExperimentRunner.calculate_recalls(ret, gt, k=0)

def test_hnsw_index_lifecycle_and_size(runner):
    """Verifies HNSW index creation, build timing, relation sizing, and catalog entry."""
    info = runner.create_hnsw_index(m=16, ef_construction=64)
    assert info["index_type"] == "HNSW"
    assert info["build_time_ms"] > 0.0
    assert info["disk_size_bytes"] > 0
    assert any(unit in info["disk_size_pretty"] for unit in ["bytes", "kB", "MB", "GB"])

    # Verify catalog presence
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE tablename = 'chunk_embeddings' AND indexname = 'idx_chunk_embeddings_hnsw_cosine';
            """)
            assert cur.fetchone()[0] == 1

def test_ivfflat_index_lifecycle_and_size(runner):
    """Verifies IVFFlat index creation, build timing, relation sizing, and catalog entry."""
    info = runner.create_ivfflat_index(lists=25)
    assert info["index_type"] == "IVFFlat"
    assert info["build_time_ms"] > 0.0
    assert info["disk_size_bytes"] > 0

    # Verify catalog presence
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM pg_indexes 
                WHERE tablename = 'chunk_embeddings' AND indexname = 'idx_chunk_embeddings_ivfflat_cosine';
            """)
            assert cur.fetchone()[0] == 1

def test_plan_verification_detects_scans(runner):
    """
    Verifies that verify_execution_plan accurately verifies physical Index Scans
    and rejects false index claims based on EXPLAIN (ANALYZE, BUFFERS).
    """
    sample_vec = runner.embedder.embed_text("hypertension and renal failure")

    # 1. Exact Sequential Scan Plan Verification
    runner.drop_vector_indexes()
    seq_session = {"enable_seqscan": "on", "enable_indexscan": "off", "enable_bitmapscan": "off"}
    is_ok, plan = runner.verify_execution_plan(None, seq_session, sample_vec, top_k=5)
    assert is_ok is True
    assert any("Seq Scan on chunk_embeddings" in line for line in plan)

    # 2. HNSW Index Scan Plan Verification
    runner.create_hnsw_index(m=16, ef_construction=64)
    hnsw_session = {"enable_seqscan": "off", "enable_indexscan": "on", "hnsw.ef_search": 40}
    is_ok, plan = runner.verify_execution_plan(runner.HNSW_INDEX_NAME, hnsw_session, sample_vec, top_k=5)
    assert is_ok is True
    assert any("Index Scan using idx_chunk_embeddings_hnsw_cosine" in line for line in plan)

    # 3. False Claim Detection: Claiming IVFFlat when only HNSW is built MUST fail verification
    is_false_claim, _ = runner.verify_execution_plan(runner.IVFFLAT_INDEX_NAME, hnsw_session, sample_vec, top_k=5)
    assert is_false_claim is False

    # 4. IVFFlat Index Scan Plan Verification
    runner.create_ivfflat_index(lists=25)
    ivf_session = {"enable_seqscan": "off", "enable_indexscan": "on", "ivfflat.probes": 5}
    is_ok, plan = runner.verify_execution_plan(runner.IVFFLAT_INDEX_NAME, ivf_session, sample_vec, top_k=5)
    assert is_ok is True
    assert any("Index Scan using idx_chunk_embeddings_ivfflat_cosine" in line for line in plan)

def test_controlled_mini_benchmark(runner):
    """
    Executes a fast end-to-end controlled benchmark on 2 queries and validates
    that all metrics and plan verifications are strictly recorded.
    """
    mini_queries = [
        "acute myocardial infarction with ST elevation",
        "type 2 diabetes mellitus with diabetic foot ulcer"
    ]
    report = runner.run_controlled_experiment(
        queries=mini_queries,
        top_k=5,
        iterations_per_query=1,
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search_values=[40],
        ivfflat_lists=25,
        ivfflat_probes_values=[5]
    )

    assert report["configurations_evaluated"] == 3  # Flat, HNSW, IVFFlat
    results = report["results"]

    # Flat check
    flat_cfg = results[0]
    assert flat_cfg["index_type"] == "Flat (Seq Scan)"
    assert flat_cfg["plan_verified"] is True
    assert flat_cfg["recall_at_k"] == 1.0

    # HNSW check
    hnsw_cfg = results[1]
    assert hnsw_cfg["index_type"] == "HNSW"
    assert hnsw_cfg["plan_verified"] is True
    assert hnsw_cfg["recall_at_k"] > 0.0
    assert hnsw_cfg["latency_ms"]["mean"] > 0.0

    # IVFFlat check
    ivf_cfg = results[2]
    assert ivf_cfg["index_type"] == "IVFFlat"
    assert ivf_cfg["plan_verified"] is True
    assert ivf_cfg["recall_at_k"] > 0.0
    assert ivf_cfg["latency_ms"]["mean"] > 0.0
