"""
Clinical RAG Execution Pipeline (Headless Academic Engine)
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. ClinicalRAGPipeline: Coordinates database retrieval, privacy verification, context formulation,
   LLM generation, and separate quality/groundedness evaluation.
2. ClinicalRAGResult: Structured result container with passages, citations, and evaluation telemetry.

*** ACADEMIC RESEARCH PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSTIC SYSTEM ***
"""

import time
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.rag.llm_provider import (
    BaseLLMProvider,
    RetrievalOnlyProvider,
    DeterministicMockLLMProvider,
    REFUSAL_PHRASE,
    get_llm_provider
)
from src.rag.evaluator import RAGEvaluator, CLINICAL_STOPWORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.pipeline")

CLINICAL_DISCLAIMER = (
    "RESEARCH PROTOTYPE ONLY: This system is strictly an academic database research prototype "
    "for evaluating relational-vector query engines. It is NOT a medical device, is NOT validated "
    "for clinical diagnosis, and must NEVER be used to make healthcare or diagnostic decisions."
)

CLINICAL_SYSTEM_PROMPT = f"""You are a specialized clinical research retrieval assistant evaluating query results from a healthcare vector database.

MANDATORY SAFETY & ACCURACY RULES:
1. {CLINICAL_DISCLAIMER}
2. Ground your answer EXCLUSIVELY in the retrieved passages provided in the prompt. Do NOT extrapolate or introduce external facts.
3. Every factual assertion MUST cite its source document ID in square brackets, e.g. [DOC-0042] or [uuid].
4. Clearly distinguish documented clinical evidence from clinical uncertainty:
   - Provide a section titled "**Documented Evidence:**" containing cited findings.
   - Provide a section titled "**Clinical Uncertainty & Limitations:**" detailing what is unrecorded, ambiguous, or cannot be inferred.
5. REFUSAL RULE: If the retrieved passages do NOT contain sufficient clinical evidence to answer the question, you MUST refuse by stating verbatim:
   "{REFUSAL_PHRASE}"
   Do NOT attempt to invent an answer when evidence is missing.
"""

@dataclass
class ClinicalRAGResult:
    """Structured response container for the clinical RAG pipeline."""
    question: str
    passages: List[Dict[str, Any]]
    answer: str
    provider_name: str
    model_name: str
    evidence_status: str
    citations: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    evaluation: Optional[Dict[str, Any]] = None
    disclaimer: str = CLINICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "evidence_status": self.evidence_status,
            "citations": self.citations,
            "passages_count": len(self.passages),
            "passages": self.passages,
            "answer": self.answer,
            "timing_ms": {
                "retrieval": round(self.retrieval_time_ms, 3),
                "generation": round(self.generation_time_ms, 3),
                "total": round(self.total_time_ms, 3)
            },
            "evaluation": self.evaluation,
            "disclaimer": self.disclaimer
        }

class ClinicalRAGPipeline:
    """
    Coordinates database retrieval from PostgreSQL/pgvector and passes retrieved
    passages to a configurable, privacy-controlled LLM provider.
    """

    def __init__(
        self,
        search_engine: Optional[PostgresHybridSearchEngine] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        relevance_threshold: float = 0.82,
        evaluator: Optional[RAGEvaluator] = None
    ):
        self.search_engine = search_engine or PostgresHybridSearchEngine()
        self.llm_provider = llm_provider or DeterministicMockLLMProvider(relevance_threshold=relevance_threshold)
        self.relevance_threshold = relevance_threshold
        self.evaluator = evaluator or RAGEvaluator(relevance_distance_threshold=relevance_threshold)

    def execute(
        self,
        question: str,
        top_k: int = 3,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        search_mode: str = "exact_flat",
        evaluate: bool = False,
        expected_doc_ids: Optional[List[str]] = None
    ) -> ClinicalRAGResult:
        """
        Executes end-to-end RAG query:
        1. Retrieves top-k document chunks via PostgreSQL with relational metadata joins.
        2. Evaluates evidence presence and semantic proximity.
        3. Formulates strictly grounded prompt containing only retrieved excerpts.
        4. Invokes configured LLM provider (or skips in retrieval-only mode).
        5. Computes decoupled retrieval quality and generation groundedness metrics if requested.
        """
        t0 = time.perf_counter()

        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only.")

        # 1. Database Retrieval Step
        t_ret_start = time.perf_counter()
        db_results = self.search_engine.search(
            query=question.strip(),
            top_k=top_k,
            department=department,
            document_type=document_type,
            min_age=min_age,
            max_age=max_age,
            search_mode=search_mode
        )
        t_ret_ms = (time.perf_counter() - t_ret_start) * 1000.0

        passages = db_results.get("results", [])

        # 2. Evidence Sufficiency Assessment
        query_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", question)
            if w.lower() not in CLINICAL_STOPWORDS
        )
        combined_text = " ".join(p.get("snippet", "") for p in passages).lower()
        matched_words = [w for w in query_words if w in combined_text]
        overlap_ratio = len(matched_words) / len(query_words) if query_words else 1.0
        best_dist = float(passages[0].get("cosine_distance", 1.0)) if passages else 1.0

        if not passages:
            evidence_status = "ZERO_DOCUMENTS"
        elif (best_dist > 0.50 and overlap_ratio < 0.50) or (best_dist > self.relevance_threshold) or (overlap_ratio == 0.0 and query_words):
            evidence_status = "INADEQUATE_EVIDENCE"
        else:
            evidence_status = "ADEQUATE_EVIDENCE"

        # 3. Context Formulation (Passing ONLY retrieved passages)
        context_blocks = []
        for idx, p in enumerate(passages, start=1):
            doc_id = p.get("synthetic_document_num") or p.get("document_id", "DOC-UNKNOWN")
            dept = p.get("department", "Unknown")
            dist = p.get("cosine_distance", 1.0)
            text = p.get("snippet", "").strip()

            context_blocks.append(
                f"[Passage {idx} | ID: {doc_id} | Department: {dept} | Cosine Distance: {dist:.4f}]\n"
                f"{text}"
            )

        context_str = "\n\n".join(context_blocks) if context_blocks else "NO DOCUMENTS RETRIEVED."

        prompt = (
            f"Question: {question.strip()}\n\n"
            f"Retrieved Clinical Documents:\n"
            f"{context_str}\n\n"
            f"Instructions:\n"
            f"Based strictly on the retrieved passages above, provide an answer. "
            f"Cite each fact with its document ID in brackets (e.g. [{passages[0].get('synthetic_document_num', 'DOC-XXXX') if passages else 'DOC-XXXX'}]). "
            f"Distinguish documented evidence from clinical uncertainty. "
            f"If the available documents do not establish an answer, state: '{REFUSAL_PHRASE}'"
        )

        # 4. LLM Generation Step
        t_gen_start = time.perf_counter()
        llm_response = self.llm_provider.generate(
            prompt=prompt,
            passages=passages,
            system_prompt=CLINICAL_SYSTEM_PROMPT
        )
        t_gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

        # 5. Decoupled Evaluation Step
        eval_report = None
        if evaluate:
            eval_report = self.evaluator.evaluate_pipeline_output(
                query=question,
                retrieved_docs=passages,
                answer=llm_response.content,
                expected_doc_ids=expected_doc_ids,
                expected_abstention=(evidence_status != "ADEQUATE_EVIDENCE")
            )

        t_total_ms = (time.perf_counter() - t0) * 1000.0

        return ClinicalRAGResult(
            question=question,
            passages=passages,
            answer=llm_response.content,
            provider_name=llm_response.provider_name,
            model_name=llm_response.model_name,
            evidence_status=evidence_status,
            citations=llm_response.citations,
            retrieval_time_ms=t_ret_ms,
            generation_time_ms=t_gen_ms,
            total_time_ms=t_total_ms,
            evaluation=eval_report,
            disclaimer=CLINICAL_DISCLAIMER
        )
