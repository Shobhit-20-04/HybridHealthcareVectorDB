"""
Automated Test Suite for Clinical RAG Pipeline & Decoupled Evaluator
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Validates:
  1. Retrieval-only mode (generation skipped, passages preserved)
  2. Data privacy guard (external LLM blocked without explicit opt-in)
  3. Missing evidence abstention protocol ("The available documents do not establish an answer")
  4. Evidence vs. uncertainty distinction in generated answers
  5. Citation validity and hallucination detection
  6. Unsupported claims detection in generation groundedness evaluator
  7. Decoupled evaluation reporting (retrieval quality vs. generation groundedness)
  8. Mandatory non-diagnostic clinical disclaimer presence
"""

import pytest
from typing import List, Dict, Any

from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.rag.llm_provider import (
    BaseLLMProvider,
    RetrievalOnlyProvider,
    DeterministicMockLLMProvider,
    ExternalAPIProvider,
    ExternalAccessBlockedError,
    REFUSAL_PHRASE,
    get_llm_provider
)
from src.rag.evaluator import (
    RetrievalQualityEvaluator,
    GenerationGroundednessEvaluator,
    RAGEvaluator
)
from src.rag.pipeline import (
    ClinicalRAGPipeline,
    CLINICAL_DISCLAIMER
)

@pytest.fixture(scope="module")
def search_engine():
    """Initializes shared PostgresHybridSearchEngine."""
    return PostgresHybridSearchEngine()

@pytest.fixture(scope="module")
def mock_pipeline(search_engine):
    """Initializes deterministic offline clinical RAG pipeline."""
    provider = DeterministicMockLLMProvider()
    return ClinicalRAGPipeline(search_engine=search_engine, llm_provider=provider)

def test_non_diagnostic_disclaimer_presence(mock_pipeline):
    """Verifies that the non-diagnostic research disclaimer is present across all responses."""
    res = mock_pipeline.execute("heart failure medications", top_k=2)
    assert "NOT a medical device" in res.disclaimer
    assert "clinical diagnosis" in res.disclaimer
    assert CLINICAL_DISCLAIMER == res.disclaimer

def test_retrieval_only_mode(search_engine):
    """Verifies that retrieval-only mode returns passages without invoking generation."""
    provider = RetrievalOnlyProvider()
    pipeline = ClinicalRAGPipeline(search_engine=search_engine, llm_provider=provider)
    
    res = pipeline.execute("chest pain and hypertension", top_k=3, evaluate=True)
    
    assert res.provider_name == "retrieval_only"
    assert len(res.passages) > 0
    assert "[RETRIEVAL-ONLY MODE] Generation skipped" in res.answer
    assert len(res.citations) > 0
    assert res.evaluation is not None
    assert res.evaluation["retrieval_quality"]["returned_count"] == len(res.passages)
    assert res.evaluation["generation_groundedness"]["refusal_correctness"].startswith("RETRIEVAL_ONLY_MODE")

def test_external_llm_opt_in_security_guard():
    """Verifies that external LLM calls are strictly blocked without explicit opt-in."""
    # 1. Direct provider instantiation without opt-in
    provider = ExternalAPIProvider(allow_external=False, api_key="test-dummy-key")
    with pytest.raises(ExternalAccessBlockedError) as exc_info:
        provider.generate("Test prompt", passages=[{"snippet": "patient data"}])
    assert "DATA PRIVACY GUARD" in str(exc_info.value)
    assert "blocked by default" in str(exc_info.value)

    # 2. Factory creation error check
    factory_provider = get_llm_provider(provider_type="external", allow_external=False)
    with pytest.raises(ExternalAccessBlockedError):
        factory_provider.generate("Test prompt", passages=[{"snippet": "patient data"}])

def test_missing_evidence_refusal(mock_pipeline):
    """
    Verifies that when no adequate evidence is retrieved, the model explicitly
    refuses by stating that the available documents do not establish an answer.
    """
    # Query outside the synthetic adult internal medicine corpus
    negative_query = "What burr hole diameter and surgical trephine were used for pediatric craniotomy?"
    res = mock_pipeline.execute(negative_query, top_k=3, evaluate=True)

    assert REFUSAL_PHRASE in res.answer
    assert "Clinical Uncertainty & Limitations:" in res.answer
    assert res.evidence_status == "INADEQUATE_EVIDENCE"
    assert len(res.citations) == 0
    
    # Verify groundedness evaluation confirms correct abstention
    assert res.evaluation is not None
    assert res.evaluation["generation_groundedness"]["refusal_correctness"] == "CORRECT_ABSTENTION"
    assert res.evaluation["generation_groundedness"]["overall_groundedness_score"] == 1.0

def test_grounded_answer_and_citations(mock_pipeline):
    """
    Verifies that when evidence is retrieved, the model cites retrieved document IDs
    and explicitly separates evidence from clinical uncertainty.
    """
    res = mock_pipeline.execute("congestive heart failure dyspnea", department="Cardiology", top_k=3, evaluate=True)

    assert res.evidence_status == "ADEQUATE_EVIDENCE"
    assert len(res.passages) > 0
    assert len(res.citations) > 0

    # Verify formatting structure
    assert "**Documented Evidence:**" in res.answer
    assert "**Clinical Uncertainty & Limitations:**" in res.answer

    # Verify all cited documents exist in retrieved passages
    retrieved_doc_ids = {
        str(p.get("synthetic_document_num", "")).upper()
        for p in res.passages
    }
    for cited_id in res.citations:
        assert cited_id.upper() in retrieved_doc_ids

    # Verify groundedness score
    assert res.evaluation is not None
    assert res.evaluation["generation_groundedness"]["citation_precision"] == 1.0
    assert res.evaluation["generation_groundedness"]["refusal_correctness"] == "PROPERLY_ANSWERED"

def test_unsupported_claim_detection():
    """Verifies that the groundedness evaluator flags fabricated claims not present in context."""
    evaluator = GenerationGroundednessEvaluator()
    passages = [
        {
            "synthetic_document_num": "SYN-DOC-0001-01-01",
            "snippet": "Patient presented with mild headache. Vitals normal. Prescribed acetaminophen 500mg as needed."
        }
    ]

    # Grounded answer
    grounded_answer = (
        "**Documented Evidence:**\n"
        "- According to [SYN-DOC-0001-01-01], patient was prescribed acetaminophen 500mg.\n\n"
        "**Clinical Uncertainty & Limitations:**\n"
        "- Follow-up imaging was not recorded."
    )
    g_metrics = evaluator.evaluate(grounded_answer, passages)
    assert len(g_metrics.unsupported_claims) == 0
    assert g_metrics.claim_groundedness_ratio == 1.0

    # Hallucinated / Unsupported answer
    hallucinated_answer = (
        "**Documented Evidence:**\n"
        "- According to [SYN-DOC-0001-01-01], patient underwent emergency craniotomy with subdural hematoma evacuation and amiodarone infusion.\n\n"
        "**Clinical Uncertainty & Limitations:**\n"
        "- Long-term neuro deficit unknown."
    )
    h_metrics = evaluator.evaluate(hallucinated_answer, passages)
    assert len(h_metrics.unsupported_claims) > 0
    assert h_metrics.claim_groundedness_ratio < 1.0

def test_invalid_citation_detection():
    """Verifies that citations of non-existent documents are flagged as invalid."""
    evaluator = GenerationGroundednessEvaluator()
    passages = [
        {"synthetic_document_num": "SYN-DOC-0001-01-01", "snippet": "Normal sinus rhythm."}
    ]

    # Citing a fabricated document ID
    fabricated_answer = (
        "**Documented Evidence:**\n"
        "- According to [SYN-DOC-9999-99-99], patient has ventricular tachycardia."
    )
    metrics = evaluator.evaluate(fabricated_answer, passages)
    assert "SYN-DOC-9999-99-99" in metrics.citations_invalid
    assert metrics.citation_precision == 0.0

def test_separate_retrieval_and_groundedness_evaluators():
    """
    Verifies that retrieval quality and generation groundedness are computed
    as decoupled, independent metric suites.
    """
    rag_evaluator = RAGEvaluator()
    passages = [
        {
            "synthetic_document_num": "SYN-DOC-0005-01-01",
            "snippet": "Type 2 diabetes managed with metformin 1000mg twice daily.",
            "cosine_distance": 0.62,
            "cosine_similarity": 0.38
        }
    ]
    answer = (
        "**Documented Evidence:**\n"
        "- According to [SYN-DOC-0005-01-01], diabetes is managed with metformin 1000mg.\n\n"
        "**Clinical Uncertainty & Limitations:**\n"
        "- HbA1c lab values not recorded."
    )

    report = rag_evaluator.evaluate_pipeline_output(
        query="metformin dosage for diabetes",
        retrieved_docs=passages,
        answer=answer,
        expected_doc_ids=["SYN-DOC-0005-01-01"]
    )

    # Retrieval Quality Suite
    rq = report["retrieval_quality"]
    assert "returned_count" in rq
    assert "top_distance" in rq
    assert "mean_distance" in rq
    assert "lexical_overlap_ratio" in rq
    assert "recall_at_k" in rq
    assert rq["returned_count"] == 1
    assert rq["recall_at_k"] == 1.0

    # Generation Groundedness Suite
    gg = report["generation_groundedness"]
    assert "citation_precision" in gg
    assert "unsupported_claims" in gg
    assert "claim_groundedness_ratio" in gg
    assert "distinguishes_uncertainty" in gg
    assert "overall_groundedness_score" in gg
    assert gg["citation_precision"] == 1.0
    assert gg["distinguishes_uncertainty"] is True
