"""
Command-Line Interface for Exact Semantic Retrieval Baseline
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes natural-language semantic similarity search using PostgreSQL and pgvector
without approximate vector indexes (ground-truth flat sequential scan).
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.exact_search import ExactSemanticSearchEngine, RetrievalError, NoEmbeddingsFoundError
from src.embeddings.text_embedder import DimensionMismatchError
from database.connection import get_db_config

def main():
    parser = argparse.ArgumentParser(
        description="Exact Semantic Similarity Search Baseline (Ground-Truth Flat Search)"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default="bilateral lower extremity edema and progressive dyspnea",
        help="Natural language clinical search query"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model ID (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Expected vector dimension (default: 384)"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Display PostgreSQL query execution plan (EXPLAIN ANALYZE) confirming sequential scan"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results for automated evaluations"
    )

    args = parser.parse_args()

    # Pre-flight check
    try:
        engine = ExactSemanticSearchEngine(
            model_id=args.model,
            expected_dimension=args.dimension
        )
        response = engine.search(
            query=args.query,
            top_k=args.top_k,
            explain=args.explain
        )
    except ValueError as e:
        print(f"\n[!] Input Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NoEmbeddingsFoundError as e:
        print(f"\n[!] Ingestion Prerequisite Error: {e}", file=sys.stderr)
        sys.exit(1)
    except DimensionMismatchError as e:
        print(f"\n[!] Model / Schema Incompatibility: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Retrieval Failure: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(response, indent=2))
        return

    cfg = get_db_config()
    print("\n" + "=" * 84)
    print("  EXACT FLAT VECTOR SEARCH ENGINE (GROUND-TRUTH BASELINE)")
    print("=" * 84)
    print(f"[*] Query String       : '{response['query']}'")
    print(f"[*] Model ID           : {response['model_id']} ({response['dimension']} dimensions)")
    print(f"[*] Search Mechanism   : Sequential Flat Scan (enable_indexscan=off, enable_bitmapscan=off)")
    print(f"[*] Corpus Evaluated   : {response['total_corpus_embeddings']} stored chunk vectors")
    print(f"[*] Top-K Requested    : {response['top_k']} (Returned: {response['returned_count']})")
    print(f"[*] Target Database    : {cfg.safe_connection_string}")
    print("-" * 84)
    print(f"  Latency Breakdown    : Embedding: {response['embedding_time_ms']:.2f} ms | "
          f"Database: {response['database_time_ms']:.2f} ms | Total: {response['total_time_ms']:.2f} ms")
    print("-" * 84)
    print("  METRIC NOTE:")
    print("    * Cosine Distance (D_C) = 1.0 - cos(theta) [Range: 0 to 2; LOWER IS CLOSER]")
    print("    * Cosine Similarity (S_C) = 1.0 - D_C      [Range: -1 to 1; HIGHER IS CLOSER]")
    print("=" * 84)

    if not response["results"]:
        print("  [!] No matching clinical passages found in the database.")
    else:
        for item in response["results"]:
            print(f"\n[Rank #{item['rank']}] Document: {item['synthetic_document_num']} | Patient: {item['synthetic_mrn']}")
            print(f"  - Title / Type       : {item['title']} ({item['document_type']})")
            print(f"  - Department         : {item['department']}")
            print(f"  - Chunk Index        : {item['chunk_index']} ({item['token_count']} tokens, {item['char_length']} chars)")
            print(f"  - Cosine Distance    : {item['cosine_distance']:.4f}  <==>  Cosine Similarity: {item['cosine_similarity']:.4f}")
            print(f"  - Clinical Snippet   :")
            # Indent snippet for clean presentation
            snippet_lines = item['snippet'].split("\n")
            for s_line in snippet_lines:
                if s_line.strip():
                    print(f"      {s_line.strip()}")

    if args.explain and response.get("query_plan"):
        print("\n" + "-" * 84)
        print("  POSTGRESQL PHYSICAL EXECUTION PLAN (EXPLAIN ANALYZE)")
        print("-" * 84)
        for line in response["query_plan"]:
            print(f"  {line}")

    print("\n" + "=" * 84 + "\n")

if __name__ == "__main__":
    main()
