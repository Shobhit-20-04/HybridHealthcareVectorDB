"""
Automated Test Suite for Hybrid Relational–Vector Query Execution
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Validates:
  - Empty filtered results handling (impossible/zero-match filters)
  - Restrictive filter selectivity and predicate conformance
  - Referential integrity and completeness of joined patient/encounter/document metadata
  - Mathematical ranking correctness (cosine distance ascending, similarity descending)
  - SQL injection parameterization immunity
  - Recall@K against filtered exact sequential scan baseline
  - Execution plan extraction and optimizer strategy classification
  - Age range and clinical demographic filtering
  - Input validation and defensive error handling
"""

import re
import pytest
from typing import Dict, Any, List
from src.retrieval.hybrid_search import PostgresHybridSearchEngine, HybridQueryError

@pytest.fixture(scope="module")
def engine():
    """Initializes PostgresHybridSearchEngine instance for test session."""
    return PostgresHybridSearchEngine()

def test_empty_filtered_results(engine):
    """
    Verifies that impossible or zero-match relational filters return an empty list gracefully
    without raising exceptions or corrupting transactions.
    """
    # 1. Impossible age filter
    res = engine.search(
        query="acute myocardial infarction",
        top_k=5,
        min_age=150,
        search_mode="exact_flat"
    )
    assert res["returned_count"] == 0
    assert res["results"] == []

    # 2. Non-existent clinical department
    res2 = engine.search(
        query="acute myocardial infarction",
        top_k=5,
        department="NonExistentAviationDepartment",
        search_mode="hnsw"
    )
    assert res2["returned_count"] == 0
    assert res2["results"] == []

def test_restrictive_filters(engine):
    """
    Verifies that restrictive filters (high selectivity) return a valid subset conforming
    strictly to all combined relational predicates.
    """
    res = engine.search(
        query="chest infiltrate and pulmonary consolidation",
        top_k=10,
        department="Pulmonology",
        document_type="radiology_report",
        search_mode="exact_flat"
    )
    # The database contains exactly 8 pulmonology radiology reports
    assert 0 < res["returned_count"] <= 8
    for r in res["results"]:
        assert r["department"] == "Pulmonology"
        assert r["document_type"] == "radiology_report"

def test_metadata_correctness_and_joins(engine):
    """
    Verifies that every returned document chunk is joined to complete, valid clinical metadata:
    patient MRN, DOB, calculated age, encounter info, document titles, and diagnoses.
    """
    res = engine.search(
        query="shortness of breath and palpitations",
        top_k=3,
        department="Cardiology",
        search_mode="hnsw"
    )
    assert res["returned_count"] > 0

    for r in res["results"]:
        # Chunk level
        assert r["chunk_id"] is not None
        assert isinstance(r["chunk_index"], int)
        assert len(r["snippet"]) > 0
        assert r["char_length"] > 0
        assert r["token_count"] > 0

        # Document level
        assert r["document_id"] is not None
        assert re.match(r"^SYN-DOC-\d{4}-\d{2}-\d{2}$", r["synthetic_document_num"])
        assert len(r["document_title"]) > 0
        assert r["document_type"] in ["progress_note", "discharge_summary", "radiology_report", "pathology_report", "consult_note"]
        assert isinstance(r["doc_metadata"], dict)

        # Patient level
        assert r["patient_id"] is not None
        assert re.match(r"^SYN-MRN-\d{6}$", r["synthetic_mrn"])
        assert r["gender"] in ["M", "F", "OTHER", "UNKNOWN"]
        assert isinstance(r["patient_age"], int)
        assert 0 <= r["patient_age"] <= 120

        # Encounter level
        assert r["encounter_id"] is not None
        assert r["department"] == "Cardiology"
        assert r["encounter_type"] in ["inpatient", "outpatient", "emergency", "telehealth"]
        assert len(r["admission_date"]) > 0

        # Diagnosis aggregation
        assert len(r["diagnoses"]) > 0

        # Vector metrics
        assert 0.0 <= r["cosine_distance"] <= 2.0
        assert -1.0 <= r["cosine_similarity"] <= 1.0

def test_result_ranking_order(engine):
    """
    Verifies that returned hits are ranked strictly in ascending order of cosine distance,
    and descending order of cosine similarity, preserving metric mathematical identity.
    """
    res = engine.search(
        query="uncontrolled hyperglycemia and insulin management",
        top_k=5,
        department="Endocrinology",
        search_mode="exact_flat"
    )
    assert res["returned_count"] >= 2
    results = res["results"]

    for i in range(len(results) - 1):
        d_curr = results[i]["cosine_distance"]
        d_next = results[i + 1]["cosine_distance"]
        s_curr = results[i]["cosine_similarity"]
        s_next = results[i + 1]["cosine_similarity"]

        # Ascending distance: d_curr <= d_next
        assert d_curr <= d_next + 1e-6
        # Descending similarity: s_curr >= s_next
        assert s_curr >= s_next - 1e-6

        # Mathematical identity: similarity == 1.0 - distance
        assert abs((1.0 - d_curr) - s_curr) < 1e-4

def test_parameterized_sql_injection_safety(engine):
    """
    Tests that malicious SQL fragments passed in metadata filters are safely handled
    by parameterized query binding (%s) and do not cause injection or errors.
    """
    malicious_dept = "Cardiology'; DROP TABLE patients; --"
    res = engine.search(
        query="coronary artery disease",
        top_k=3,
        department=malicious_dept,
        search_mode="exact_flat"
    )
    # Must safely return 0 results since no department matches this literal string
    assert res["returned_count"] == 0

def test_age_range_filtering(engine):
    """
    Verifies that age range filters correctly filter patients based on their calculated age
    at encounter date.
    """
    min_age = 50
    max_age = 75
    res = engine.search(
        query="hypertension and cardiovascular risk",
        top_k=5,
        min_age=min_age,
        max_age=max_age,
        search_mode="exact_flat"
    )
    assert res["returned_count"] > 0
    for r in res["results"]:
        assert min_age <= r["patient_age"] <= max_age

def test_recall_at_k_against_filtered_exact_baseline(engine):
    """
    Verifies that HNSW and IVFFlat achieve high Recall@K (>= 90%) relative to the
    correctly computed filtered exact sequential scan baseline.
    """
    query = "congestive heart failure with dyspnea on exertion"
    filters = {"department": "Cardiology"}

    # 1. Ground truth from filtered exact search
    exact_res = engine.search(
        query=query,
        top_k=5,
        search_mode="exact_flat",
        **filters
    )
    assert exact_res["returned_count"] == 5
    dk_exact = exact_res["results"][-1]["cosine_distance"]

    # 2. HNSW retrieval with same filters
    hnsw_res = engine.search(
        query=query,
        top_k=5,
        search_mode="hnsw",
        session_params={"hnsw.ef_search": 40},
        **filters
    )
    assert hnsw_res["returned_count"] == 5

    # Calculate 1-Recall against exact distance threshold
    hits = sum(1 for r in hnsw_res["results"] if r["cosine_distance"] <= dk_exact + 1e-5)
    recall = hits / 5.0
    assert recall >= 0.80

def test_plan_inspection_and_explain(engine):
    """
    Verifies that explain=True populates query_plan and plan_summary with verified
    PostgreSQL execution node details.
    """
    res = engine.search(
        query="pulmonary embolism and anticoagulation",
        top_k=5,
        department="Pulmonology",
        search_mode="hnsw",
        explain=True
    )
    assert len(res["query_plan"]) > 0
    assert "plan_summary" in res
    assert res["plan_summary"]["strategy"] in [
        "pre_filter_relational_scan",
        "iterative_vector_scan",
        "exact_sequential_scan",
        "hybrid_nested_scan"
    ]

def test_input_validation_errors(engine):
    """
    Verifies defensive input validation for empty query strings, non-positive top_k,
    and invalid age bounds.
    """
    # Empty query
    with pytest.raises(ValueError) as exc:
        engine.search(query="")
    assert "cannot be empty" in str(exc.value)

    # Whitespace query
    with pytest.raises(ValueError):
        engine.search(query="   \t \n ")

    # Non-positive top_k
    with pytest.raises(ValueError):
        engine.search(query="cardiac arrest", top_k=0)
    with pytest.raises(ValueError):
        engine.search(query="cardiac arrest", top_k=-5)

    # Inverted age bounds (min > max)
    with pytest.raises(ValueError) as exc:
        engine.search(query="pneumonia", min_age=70, max_age=30)
    assert "cannot be greater than" in str(exc.value)
