"""
Command-Line Clinical RAG Demonstration Interface
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

A headless, command-line demonstration of retrieval-augmented generation grounded in a PostgreSQL/pgvector
database. Prominently displays non-diagnostic research disclaimers, enforces opt-in privacy guards,
displays retrieved document IDs and excerpts, and evaluates retrieval quality separately from generation groundedness.

NO WEB INTERFACE OR CHATBOT UI: Strictly academic terminal execution.
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.rag.llm_provider import (
    get_llm_provider,
    ExternalAccessBlockedError,
    REFUSAL_PHRASE
)
from src.rag.pipeline import (
    ClinicalRAGPipeline,
    CLINICAL_DISCLAIMER
)

# Standard sample demonstration queries
SAMPLE_QUERIES = [
    {
        "title": "Query 1: Supported Clinical Condition (Cardiology / Heart Failure)",
        "question": "What medications and lifestyle restrictions are documented for congestive heart failure and dyspnea?",
        "department": "Cardiology",
        "top_k": 3,
        "description": "Tests grounded synthesis from matching cardiology records with explicit document citations."
    },
    {
        "title": "Query 2: High-Selectivity Department Filter (Pulmonology)",
        "question": "What respiratory findings, lung sounds, and inhaler therapies were noted?",
        "department": "Pulmonology",
        "top_k": 3,
        "description": "Tests hybrid relational pre-filtering where only pulmonology notes are retrieved."
    },
    {
        "title": "Query 3: Missing Evidence / Negative Query (Pediatric Neurosurgery)",
        "question": "What surgical protocol and burr hole sizes were utilized for pediatric craniotomy?",
        "department": None,
        "top_k": 3,
        "description": "Tests mandatory refusal rule: corpus contains adult internal medicine, so model MUST state that available documents do not establish an answer."
    }
]

def print_banner():
    print("=" * 82)
    print("  HYBRID RELATIONAL-VECTOR CLINICAL RAG DEMONSTRATION")
    print("  Database System: PostgreSQL 18.6 + pgvector v0.8.6 | Embeddings: 384-d Cosine")
    print("=" * 82)
    print(f"  [!] {CLINICAL_DISCLAIMER}")
    print("=" * 82)

def display_rag_result(result, evaluate: bool = False, json_output: bool = False):
    """Renders formatted CLI output for a single RAG execution."""
    if json_output:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("\n" + "-" * 82)
    print(f"QUESTION: {result.question}")
    print(f"PROVIDER: {result.provider_name} (Model: {result.model_name})")
    print(f"EVIDENCE STATUS: {result.evidence_status}")
    print(f"TIMING: Retrieval = {result.retrieval_time_ms:.1f}ms | Generation = {result.generation_time_ms:.1f}ms | Total = {result.total_time_ms:.1f}ms")
    print("-" * 82)

    # 1. Retrieved Document Passages
    print("\n>>> RETRIEVED DATABASE PASSAGES (Top-K Chunks):")
    if not result.passages:
        print("    [!] Zero qualifying document chunks returned by database.")
    else:
        for idx, p in enumerate(result.passages, start=1):
            doc_id = p.get("synthetic_document_num") or p.get("document_id")
            dept = p.get("department", "General")
            doc_type = p.get("document_type", "note")
            dist = p.get("cosine_distance", 1.0)
            sim = p.get("cosine_similarity", 1.0 - dist)
            snippet = p.get("snippet", "").strip()
            
            # Truncate snippet preview cleanly for terminal readability
            preview = snippet if len(snippet) <= 220 else snippet[:217] + "..."

            print(f"    [{idx}] ID: {doc_id:<10} | Dept: {dept:<16} | Type: {doc_type:<18} | Dist: {dist:.4f} (Sim: {sim:.4f})")
            print(f"        Excerpt: \"{preview}\"")

    # 2. Generated Response
    print("\n>>> GENERATED RESPONSE:")
    print(result.answer)

    # 3. Cited Identifiers
    if result.citations:
        print(f"\n>>> CITED DOCUMENT IDENTIFIERS: {', '.join(result.citations)}")
    else:
        print("\n>>> CITED DOCUMENT IDENTIFIERS: None (Abstention or Zero Evidence)")

    # 4. Decoupled Evaluation Report
    if evaluate and result.evaluation:
        ev = result.evaluation
        rq = ev["retrieval_quality"]
        gg = ev["generation_groundedness"]

        print("\n" + "=" * 82)
        print("  DECOUPLED ACADEMIC EVALUATION REPORT")
        print("=" * 82)
        print("  [PART 1: RETRIEVAL QUALITY (Database Search Performance)]")
        print(f"    - Returned Passages       : {rq['returned_count']}")
        print(f"    - Best Cosine Distance    : {rq['top_distance']} (Cosine Sim: {rq['top_similarity']})")
        print(f"    - Mean Cosine Distance    : {rq['mean_distance']}")
        print(f"    - Query Lexical Overlap   : {rq['lexical_overlap_ratio'] * 100:.1f}% (Matched: {rq['matched_query_keywords']})")
        print(f"    - Adequate Evidence Flag  : {rq['has_adequate_evidence']}")
        if rq.get("recall_at_k") is not None:
            print(f"    - Recall@{len(result.passages)} (Ground Truth) : {rq['recall_at_k'] * 100:.1f}%")

        print("\n  [PART 2: GENERATION GROUNDEDNESS (LLM Verification)]")
        print(f"    - Citations Checked       : {gg['citations_total']} total ({len(gg['citations_valid'])} valid, {len(gg['citations_invalid'])} invalid)")
        print(f"    - Citation Precision      : {gg['citation_precision'] * 100:.1f}%")
        print(f"    - Refusal Protocol Status : {gg['refusal_correctness']}")
        print(f"    - Unsupported Claims      : {len(gg['unsupported_claims'])} detected")
        if gg["unsupported_claims"]:
            for uc in gg["unsupported_claims"]:
                print(f"        * Hallucination Alert: \"{uc}\"")
        print(f"    - Claim Groundedness Ratio: {gg['claim_groundedness_ratio'] * 100:.1f}%")
        print(f"    - Distinguishes Uncertainty: {gg['distinguishes_uncertainty']}")
        print(f"    - Overall Groundedness Score: {gg['overall_groundedness_score']:.2f} / 1.00")
        print("=" * 82)

def run_single_query(pipeline, query: str, args):
    """Executes a single user-specified query through the pipeline."""
    res = pipeline.execute(
        question=query,
        top_k=args.top_k,
        department=args.department,
        document_type=args.document_type,
        min_age=args.min_age,
        max_age=args.max_age,
        search_mode=args.search_mode,
        evaluate=args.evaluate
    )
    display_rag_result(res, evaluate=args.evaluate, json_output=args.json)

def run_sample_suite(pipeline, args):
    """Runs the 3 curated clinical demonstration cases."""
    print("\n[+] Running standard clinical RAG demonstration suite (3 cases)...")
    for item in SAMPLE_QUERIES:
        print("\n" + "#" * 82)
        print(f"  {item['title']}")
        print(f"  Objective: {item['description']}")
        print("#" * 82)

        res = pipeline.execute(
            question=item["question"],
            top_k=item["top_k"],
            department=item["department"],
            search_mode=args.search_mode,
            evaluate=True
        )
        display_rag_result(res, evaluate=True, json_output=args.json)

def main():
    parser = argparse.ArgumentParser(
        description="Command-Line Clinical RAG Demonstration (Hybrid Relational-Vector Database)"
    )
    parser.add_argument("query", nargs="?", default=None, help="Clinical natural language query")
    parser.add_argument("--top-k", type=int, default=3, help="Number of document chunks to retrieve (default: 3)")
    parser.add_argument("--department", type=str, default=None, help="Relational department filter (e.g. Cardiology, Pulmonology)")
    parser.add_argument("--document-type", type=str, default=None, help="Relational document type filter (e.g. discharge_summary, progress_note)")
    parser.add_argument("--min-age", type=int, default=None, help="Minimum patient age filter")
    parser.add_argument("--max-age", type=int, default=None, help="Maximum patient age filter")
    parser.add_argument("--provider", type=str, default="mock", choices=["mock", "retrieval-only", "ollama", "external"],
                        help="LLM provider: 'mock' (offline deterministic, default), 'retrieval-only', 'ollama' (local), 'external' (opt-in cloud)")
    parser.add_argument("--allow-external", action="store_true", help="Explicit opt-in required to allow sending passages to external LLM APIs")
    parser.add_argument("--model", type=str, default=None, help="Custom LLM model identifier")
    parser.add_argument("--search-mode", type=str, default="exact_flat", choices=["exact_flat", "hnsw", "ivfflat", "planner_choice"],
                        help="Database search mode (default: exact_flat)")
    parser.add_argument("--evaluate", action="store_true", help="Run decoupled retrieval quality and generation groundedness evaluation")
    parser.add_argument("--sample-suite", action="store_true", help="Execute all 3 clinical demonstration test cases")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    print_banner()

    # Opt-in privacy notification
    if args.provider == "external" and not args.allow_external:
        print("\n[!] SECURITY ERROR: External LLM access requires explicit opt-in.")
        print("    Clinical passages will NOT be transmitted to an external service.")
        print("    Pass --allow-external if you explicitly choose to enable external API access.\n")
        sys.exit(1)

    try:
        provider = get_llm_provider(
            provider_type=args.provider,
            allow_external=args.allow_external,
            model_name=args.model
        )
    except ExternalAccessBlockedError as e:
        print(f"\n[!] DATA PRIVACY GUARD TRIGGERED: {e}")
        sys.exit(1)

    pipeline = ClinicalRAGPipeline(
        search_engine=PostgresHybridSearchEngine(),
        llm_provider=provider
    )

    if args.sample_suite:
        run_sample_suite(pipeline, args)
    elif args.query:
        run_single_query(pipeline, args.query, args)
    else:
        # Default behavior: run sample suite so user sees immediate results
        print("\nNo specific query provided. Executing demonstration suite with evaluation...")
        args.evaluate = True
        run_sample_suite(pipeline, args)

if __name__ == "__main__":
    main()
