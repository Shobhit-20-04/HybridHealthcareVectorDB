"""
Clinical RAG Evaluation Framework: Separate Retrieval Quality & Generation Groundedness
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. RetrievalQualityEvaluator: Measures database retrieval quality (Top-1 Distance, Mean Distance, Lexical Overlap, Recall@K)
2. GenerationGroundednessEvaluator: Evaluates LLM generation groundedness (Citation Validity, Unsupported Claim Detection, Refusal Correctness)
3. RAGEvaluator: Consolidated academic evaluation engine
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from src.rag.llm_provider import REFUSAL_PHRASE

CLINICAL_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "of", "to", "for", "with", "on", "at",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "this", "that", "these", "those", "how",
    "patient", "clinical", "note", "medical", "report", "evaluation", "shows",
    "findings", "history", "assessment", "plan", "denies", "reports", "stated",
    "used", "use", "using", "performed", "cases", "case", "also", "table"
}

@dataclass
class RetrievalQualityMetrics:
    """Quantitative metrics evaluating database retrieval effectiveness."""
    query: str
    returned_count: int
    top_distance: float
    mean_distance: float
    top_similarity: float
    lexical_overlap_ratio: float
    matched_query_keywords: List[str] = field(default_factory=list)
    has_adequate_evidence: bool = False
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "returned_count": self.returned_count,
            "top_distance": round(self.top_distance, 4),
            "mean_distance": round(self.mean_distance, 4),
            "top_similarity": round(self.top_similarity, 4),
            "lexical_overlap_ratio": round(self.lexical_overlap_ratio, 4),
            "matched_query_keywords": self.matched_query_keywords,
            "has_adequate_evidence": self.has_adequate_evidence,
            "recall_at_k": round(self.recall_at_k, 4) if self.recall_at_k is not None else None,
            "precision_at_k": round(self.precision_at_k, 4) if self.precision_at_k is not None else None,
        }

@dataclass
class GroundednessMetrics:
    """Quantitative metrics evaluating LLM response groundedness against retrieved context."""
    citations_total: int
    citations_valid: List[str] = field(default_factory=list)
    citations_invalid: List[str] = field(default_factory=list)
    citation_precision: float = 1.0
    refusal_correctness: str = "UNKNOWN"
    unsupported_claims: List[str] = field(default_factory=list)
    supported_claims_count: int = 0
    claim_groundedness_ratio: float = 1.0
    distinguishes_uncertainty: bool = False
    overall_groundedness_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations_total": self.citations_total,
            "citations_valid": self.citations_valid,
            "citations_invalid": self.citations_invalid,
            "citation_precision": round(self.citation_precision, 4),
            "refusal_correctness": self.refusal_correctness,
            "unsupported_claims": self.unsupported_claims,
            "supported_claims_count": self.supported_claims_count,
            "claim_groundedness_ratio": round(self.claim_groundedness_ratio, 4),
            "distinguishes_uncertainty": self.distinguishes_uncertainty,
            "overall_groundedness_score": round(self.overall_groundedness_score, 4),
        }

class RetrievalQualityEvaluator:
    """Evaluates the search engine retrieval output independently of generation."""

    def __init__(self, relevance_distance_threshold: float = 0.82):
        self.relevance_threshold = relevance_distance_threshold

    def evaluate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        expected_doc_ids: Optional[List[str]] = None
    ) -> RetrievalQualityMetrics:
        query_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", query)
            if w.lower() not in CLINICAL_STOPWORDS
        )

        if not retrieved_docs:
            return RetrievalQualityMetrics(
                query=query,
                returned_count=0,
                top_distance=1.0,
                mean_distance=1.0,
                top_similarity=0.0,
                lexical_overlap_ratio=0.0,
                matched_query_keywords=[],
                has_adequate_evidence=False,
                recall_at_k=0.0 if expected_doc_ids else None,
                precision_at_k=0.0 if expected_doc_ids else None
            )

        top_dist = float(retrieved_docs[0].get("cosine_distance", 1.0))
        mean_dist = sum(float(d.get("cosine_distance", 1.0)) for d in retrieved_docs) / len(retrieved_docs)
        top_sim = float(retrieved_docs[0].get("cosine_similarity", 1.0 - top_dist))

        # Check lexical overlap with query terms across retrieved snippets
        combined_text = " ".join(d.get("snippet", "") for d in retrieved_docs).lower()
        matched_words = [w for w in query_words if w in combined_text]
        overlap_ratio = len(matched_words) / len(query_words) if query_words else 1.0

        # Adequate evidence requires either low cosine distance or strong term overlap
        has_evidence = (top_dist <= self.relevance_threshold) and (overlap_ratio >= 0.50 or top_dist < 0.50) and (overlap_ratio > 0.0 or not query_words)

        # Ground truth recall if known
        rec_k = None
        prec_k = None
        if expected_doc_ids:
            retrieved_ids = set()
            for d in retrieved_docs:
                if d.get("synthetic_document_num"):
                    retrieved_ids.add(str(d["synthetic_document_num"]).strip().upper())
                if d.get("document_id"):
                    retrieved_ids.add(str(d["document_id"]).strip().upper())

            exp_set = set(str(e).strip().upper() for e in expected_doc_ids)
            intersection = retrieved_ids.intersection(exp_set)
            rec_k = len(intersection) / len(exp_set) if exp_set else 1.0
            prec_k = len(intersection) / len(retrieved_docs) if retrieved_docs else 0.0

        return RetrievalQualityMetrics(
            query=query,
            returned_count=len(retrieved_docs),
            top_distance=top_dist,
            mean_distance=mean_dist,
            top_similarity=top_sim,
            lexical_overlap_ratio=overlap_ratio,
            matched_query_keywords=matched_words,
            has_adequate_evidence=has_evidence,
            recall_at_k=rec_k,
            precision_at_k=prec_k
        )

class GenerationGroundednessEvaluator:
    """Evaluates the language model's output for evidence groundedness, hallucination, and citation accuracy."""

    def evaluate(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        expected_abstention: bool = False
    ) -> GroundednessMetrics:
        if not answer or not answer.strip():
            return GroundednessMetrics(
                citations_total=0,
                citation_precision=0.0,
                refusal_correctness="EMPTY_RESPONSE",
                overall_groundedness_score=0.0
            )

        if "[RETRIEVAL-ONLY MODE]" in answer:
            ret_ids = [
                str(d.get("synthetic_document_num") or d.get("document_id"))
                for d in retrieved_docs
                if d.get("synthetic_document_num") or d.get("document_id")
            ]
            return GroundednessMetrics(
                citations_total=len(ret_ids),
                citations_valid=ret_ids,
                citations_invalid=[],
                citation_precision=1.0,
                refusal_correctness="RETRIEVAL_ONLY_MODE (Generation intentionally skipped)",
                unsupported_claims=[],
                supported_claims_count=0,
                claim_groundedness_ratio=1.0,
                distinguishes_uncertainty=True,
                overall_groundedness_score=1.0
            )

        # 1. Collect valid document IDs from retrieved passages
        valid_ids: Set[str] = set()
        passages_text_corpus = ""
        for d in retrieved_docs:
            if d.get("synthetic_document_num"):
                valid_ids.add(str(d["synthetic_document_num"]).strip().upper())
            if d.get("document_id"):
                valid_ids.add(str(d["document_id"]).strip().lower())
            passages_text_corpus += " " + d.get("snippet", "")

        passages_text_lower = passages_text_corpus.lower()

        # 2. Citation Extraction & Verification
        citation_pattern = r"\[((?:PMC-DOC|SYN-DOC|DOC)-[A-Za-z0-9\-]+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
        raw_citations = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        valid_citations = []
        invalid_citations = []
        for c in raw_citations:
            norm = c.upper() if (c.upper().startswith("DOC-") or c.upper().startswith("SYN-DOC-") or c.upper().startswith("PMC-DOC-")) else c.lower()
            if norm in valid_ids:
                if norm not in valid_citations:
                    valid_citations.append(norm)
            else:
                if norm not in invalid_citations:
                    invalid_citations.append(norm)

        total_citations = len(raw_citations)
        if total_citations > 0:
            citation_precision = len(valid_citations) / float(len(valid_citations) + len(invalid_citations))
        else:
            # If no citations were made, 1.0 only if refusal occurred; 0.0 if factual claims made without citing
            citation_precision = 1.0 if (REFUSAL_PHRASE.lower() in answer.lower() or not retrieved_docs) else 0.0

        # 3. Refusal Correctness Check
        is_refusal = REFUSAL_PHRASE.lower() in answer.lower()
        if expected_abstention:
            if is_refusal:
                refusal_status = "CORRECT_ABSTENTION"
            else:
                refusal_status = "UNGROUNDED_HALLUCINATION (Expected Refusal)"
        else:
            if is_refusal:
                refusal_status = "UNNECESSARY_REFUSAL"
            else:
                refusal_status = "PROPERLY_ANSWERED"

        # 4. Uncertainty & Limitations Distinction
        uncertainty_markers = [
            "uncertainty", "limitations", "unsubstantiated", "cannot be inferred",
            "not established", "restricted to", "not recorded"
        ]
        distinguishes_uncertainty = any(m in answer.lower() for m in uncertainty_markers)

        # 5. Unsupported Claims & Hallucination Detection
        unsupported_claims: List[str] = []
        supported_claims_count = 0

        if not is_refusal:
            # Extract factual assertions specifically from the evidence section (exclude uncertainty notes)
            evidence_text = answer
            if "**Clinical Uncertainty" in answer:
                evidence_text = answer.split("**Clinical Uncertainty")[0]
            elif "Clinical Uncertainty & Limitations:" in answer:
                evidence_text = answer.split("Clinical Uncertainty & Limitations:")[0]

            # Split into individual lines and sentences, stripping out markdown headers
            raw_lines = [
                line.strip() for line in evidence_text.splitlines()
                if line.strip() and not any(header in line.lower() for header in ["documented evidence:", "instructions:"])
            ]
            sentences = []
            for line in raw_lines:
                clean_line = re.sub(r"^[-*•]\s*", "", line)
                sub_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_line) if s.strip()]
                sentences.extend(sub_sents if sub_sents else [clean_line])

            for sent in sentences:
                # Extract clinical terms and numeric values (dosages, measurements, vitals)
                tokens = re.findall(r"\b[a-zA-Z]{4,}\b|\b\d+(?:\.\d+)?(?:\s*(?:mg|dl|mmhg|bpm|g|kg|ml))?\b", sent, re.IGNORECASE)
                content_tokens = [t.lower() for t in tokens if t.lower() not in CLINICAL_STOPWORDS]

                if not content_tokens:
                    continue

                # Check if majority of informative tokens in this sentence exist in retrieved passages
                matched = [t for t in content_tokens if t in passages_text_lower]
                coverage = len(matched) / len(content_tokens)

                if coverage < 0.40:
                    unsupported_claims.append(sent)
                else:
                    supported_claims_count += 1

            total_claims = supported_claims_count + len(unsupported_claims)
            claim_groundedness_ratio = (supported_claims_count / total_claims) if total_claims > 0 else 1.0
        else:
            claim_groundedness_ratio = 1.0

        # 6. Overall Groundedness Score Calculation
        if refusal_status == "CORRECT_ABSTENTION":
            overall_score = 1.0
        elif refusal_status.startswith("UNGROUNDED_HALLUCINATION"):
            overall_score = 0.0
        else:
            score = (
                0.40 * citation_precision +
                0.40 * claim_groundedness_ratio +
                0.20 * (1.0 if distinguishes_uncertainty else 0.0)
            )
            overall_score = max(0.0, min(1.0, score))

        return GroundednessMetrics(
            citations_total=total_citations,
            citations_valid=valid_citations,
            citations_invalid=invalid_citations,
            citation_precision=citation_precision,
            refusal_correctness=refusal_status,
            unsupported_claims=unsupported_claims,
            supported_claims_count=supported_claims_count,
            claim_groundedness_ratio=claim_groundedness_ratio,
            distinguishes_uncertainty=distinguishes_uncertainty,
            overall_groundedness_score=overall_score
        )

class RAGEvaluator:
    """Master academic evaluation harness combining separate retrieval and groundedness evaluators."""

    def __init__(self, relevance_distance_threshold: float = 0.82):
        self.retrieval_evaluator = RetrievalQualityEvaluator(relevance_distance_threshold)
        self.groundedness_evaluator = GenerationGroundednessEvaluator()

    def evaluate_pipeline_output(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        expected_doc_ids: Optional[List[str]] = None,
        expected_abstention: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Executes complete decoupled evaluation:
        1. Retrieval quality evaluated strictly against database results.
        2. Generation groundedness evaluated strictly against passages and refusal rules.
        """
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            expected_doc_ids=expected_doc_ids
        )

        should_abstain = expected_abstention if expected_abstention is not None else (not retrieval_metrics.has_adequate_evidence)

        groundedness_metrics = self.groundedness_evaluator.evaluate(
            answer=answer,
            retrieved_docs=retrieved_docs,
            expected_abstention=should_abstain
        )

        return {
            "retrieval_quality": retrieval_metrics.to_dict(),
            "generation_groundedness": groundedness_metrics.to_dict(),
            "disclaimer": "Academic prototype evaluation metrics only. NOT validated for clinical decision-making."
        }
