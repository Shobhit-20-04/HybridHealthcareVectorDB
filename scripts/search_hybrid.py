"""
Ad-Hoc Hybrid Relational–Vector Semantic Search CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes parameterized hybrid queries combining SQL relational metadata predicates
(department, document type, age ranges, ICD-10 codes) with dense vector similarity ranking.
Projects full patient, encounter, document, and diagnosis metadata alongside cosine metrics.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_search import PostgresHybridSearchEngine

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Relational–Vector Clinical Retrieval CLI"
    )
    parser.add_argument("query", type=str, help="Clinical natural language query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest documents to return (default: 5)")
    parser.add_argument("--department", type=str, default=None, help="Relational filter: Clinical department")
    parser.add_argument("--document-type", type=str, default=None, help="Relational filter: Clinical document type")
    parser.add_argument("--min-age", type=int, default=None, help="Relational filter: Minimum patient age at encounter")
    parser.add_argument("--max-age", type=int, default=None, help="Relational filter: Maximum patient age at encounter")
    parser.add_argument("--icd10-code", type=str, default=None, help="Relational filter: Synthetic ICD-10 diagnosis code")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["exact_flat", "hnsw", "ivfflat", "planner_choice"], 
        default="hnsw",
        help="Retrieval execution mode (default: hnsw)"
    )
    parser.add_argument("--explain", action="store_true", help="Print PostgreSQL EXPLAIN (ANALYZE, BUFFERS) execution plan")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    engine = PostgresHybridSearchEngine()

    response = engine.search(
        query=args.query,
        top_k=args.top_k,
        department=args.department,
        document_type=args.document_type,
        min_age=args.min_age,
        max_age=args.max_age,
        icd10_code=args.icd10_code,
        search_mode=args.mode,
        explain=args.explain
    )

    if args.json:
        print(json.dumps(response, indent=2))
        return

    print("\n" + "=" * 90)
    print("  HYBRID RELATIONAL–VECTOR CLINICAL INFORMATION RETRIEVAL")
    print("=" * 90)
    print(f"[*] Query String     : '{response['query']}'")
    print(f"[*] Search Mode      : {response['search_mode'].upper()}")
    print(f"[*] Embedding Model  : {response['model_id']} ({response['dimension']}-dim)")
    print(f"[*] Top-K Requested  : {response['top_k']} (Returned: {response['returned_count']})")
    print(f"[*] Metadata Filters : {json.dumps(response['filters_applied'])}")
    print(f"[*] Latency Metrics  : Embed={response['embedding_time_ms']}ms | DB={response['database_time_ms']}ms | Total={response['total_time_ms']}ms")

    if response.get("plan_summary"):
        print(f"[*] Optimizer Strategy: {response['plan_summary'].get('strategy', 'N/A')}")

    print("-" * 90)

    if not response["results"]:
        print("\n[!] No matching records found matching both relational filters and semantic criteria.")
    else:
        for r in response["results"]:
            print(f"\n[Rank #{r['rank']}] Cosine Distance: {r['cosine_distance']:.4f} | Cosine Similarity: {r['cosine_similarity']:.4f}")
            print(f"    Document   : {r['document_title']} ({r['document_type']}) [Doc #: {r['synthetic_document_num']}]")
            print(f"    Patient    : MRN: {r['synthetic_mrn']} | Age: {r['patient_age']} | Gender: {r['gender']} | DOB: {r['date_of_birth']}")
            print(f"    Encounter  : Dept: {r['department']} | Type: {r['encounter_type']} | Date: {r['admission_date']}")
            print(f"    Diagnoses  : {r['diagnoses']}")
            print(f"    Passage    : {r['snippet'][:140]}...")

    if args.explain and response.get("query_plan"):
        print("\n" + "=" * 90)
        print("  POSTGRESQL EXPLAIN (ANALYZE, BUFFERS) EXECUTION PLAN")
        print("=" * 90)
        for line in response["query_plan"]:
            print("  " + line)

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
