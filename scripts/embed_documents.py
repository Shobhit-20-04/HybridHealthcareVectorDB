"""
Clinical Document Embedding & Vector Storage CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes end-to-end vector pipeline:
1. Document Preprocessing & Chunking (retaining source IDs & metadata)
2. Model & Dimension Verification against pg_attribute
3. Batch Embedding Generation using Sentence Transformers
4. Idempotent Vector Storage in PostgreSQL via pgvector
5. Embedding Reconciliation Audit & Semantic Retrieval Smoke Test
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError
from src.embeddings.chunker import ClinicalDocumentChunker
from src.embeddings.vector_store import PgVectorStore
from database.connection import get_connection, get_db_config

def run_smoke_test(store: PgVectorStore) -> bool:
    """
    Executes a quick semantic similarity smoke test:
    Queries the database using a clinical phrase and checks cosine distance ranking.
    """
    sample_query = "dyspnea on exertion and pulmonary edema"
    print("\n" + "=" * 78)
    print("  RUNNING SEMANTIC RETRIEVAL SMOKE TEST")
    print("=" * 78)
    print(f"[*] Query String       : '{sample_query}'")
    print(f"[*] Embedding Model    : {store.model_id}")

    try:
        query_vec = store.embedder.embed_text(sample_query)
        store.embedder.validate_dimension(query_vec)

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Query nearest chunk using pgvector cosine distance operator <=>
                cur.execute("""
                    SELECT 
                        ce.chunk_id,
                        dc.chunk_index,
                        cd.title,
                        cd.document_type,
                        dc.chunk_text,
                        (ce.embedding <=> %s::vector) AS cosine_distance
                    FROM chunk_embeddings ce
                    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    JOIN clinical_documents cd ON dc.document_id = cd.document_id
                    WHERE ce.model_id = %s
                    ORDER BY ce.embedding <=> %s::vector
                    LIMIT 1;
                """, (query_vec, store.model_id, query_vec))
                
                match = cur.fetchone()

        if match:
            chunk_id, idx, title, doc_type, text, dist = match
            sim = 1.0 - dist
            print(f"[V] Nearest Match Found:")
            print(f"    - Chunk ID         : {chunk_id}")
            print(f"    - Document Title   : {title} ({doc_type})")
            print(f"    - Cosine Distance  : {dist:.4f} (Cosine Similarity: {sim:.4f})")
            print(f"    - Text Snippet     : {text[:140]}...")
            print("[V] Smoke test PASSED: pgvector semantic distance operator (<=>) verified.")
            return True
        else:
            print("[!] Warning: No chunk embeddings found in database to evaluate.")
            return False

    except Exception as e:
        print(f"[!] Smoke Test FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Clinical Document Preprocessing, Chunking, and pgvector Embedding Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Vector embedding dimension (default: 384)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for inference (default: auto)"
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Force re-chunking of all clinical documents in PostgreSQL"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generating embeddings for all chunks even if already present"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run semantic retrieval smoke test after embedding generation"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only display database counts and invariant status without generating embeddings"
    )

    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  CLINICAL VECTOR EMBEDDING & STORAGE PIPELINE")
    print("=" * 78)
    print(f"[*] Embedding Model  : {args.model}")
    print(f"[*] Target Dimension : {args.dimension}")
    print(f"[*] Batch Size       : {args.batch_size}")
    print(f"[*] Compute Device   : {args.device}")

    cfg = get_db_config()
    print(f"[*] Target Database  : {cfg.safe_connection_string}")

    # 1. Initialize embedder and vector store
    embedder = TextEmbedder(model_name=args.model, dimension=args.dimension, device=args.device)
    store = PgVectorStore(embedder=embedder, model_id=args.model, dimension=args.dimension)

    if args.verify_only:
        counts = store.get_embedding_counts()
        print("\n" + "-" * 78)
        print("  EMBEDDING STATUS & COVERAGE REPORT")
        print("-" * 78)
        print(f"  Total Clinical Documents : {counts['total_documents']:>6}")
        print(f"  Total Document Chunks    : {counts['total_chunks']:>6}")
        print(f"  Model Embeddings ({args.dimension}-d)  : {counts['model_embeddings']:>6}")
        print(f"  Coverage Percentage      : {counts['coverage_pct']:>6.1f}%")
        print(f"  Un-embedded Chunks       : {counts['unembedded_chunks']:>6}")
        print(f"  Orphan Embeddings        : {counts['orphan_embeddings']:>6}")
        print("=" * 78 + "\n")
        return

    # 2. Document Preprocessing & Chunking
    print("\n[1] Checking document chunking status...")
    chunker = ClinicalDocumentChunker()
    chunk_stats = chunker.process_and_store_chunks(force_rechunk=args.rechunk)
    if chunk_stats["documents_processed"] > 0:
        print(f"[+] Processed {chunk_stats['documents_processed']} docs -> Stored {chunk_stats['chunks_stored']} chunks.")
    else:
        print("[V] All documents in PostgreSQL already have valid chunks.")

    # 3. Model Registration & Column Verification
    print("\n[2] Verifying model registry and PostgreSQL vector column dimensions...")
    try:
        store.check_and_register_model()
        store.verify_column_dimension("chunk_embeddings", "embedding")
    except DimensionMismatchError as e:
        print(f"\n[!] DIMENSION MISMATCH ERROR: {e}")
        sys.exit(1)

    # 4. Batch Vector Generation & Storage
    print("\n[3] Generating dense vectors and storing into PostgreSQL via pgvector...")
    start_time = time.time()
    try:
        ingest_stats = store.ingest_chunk_embeddings(batch_size=args.batch_size, force=args.force)
        duration = time.time() - start_time
    except Exception as e:
        print(f"\n[!] Ingestion Failure: {e}")
        sys.exit(1)

    # 5. Final Reconciliation Audit
    counts = store.get_embedding_counts()
    print("\n" + "-" * 78)
    print("  INGESTION EXECUTION & RECONCILIATION SUMMARY")
    print("-" * 78)
    print(f"  Chunks Processed in Run  : {ingest_stats['chunks_processed']:>6}")
    print(f"  Embeddings Upserted      : {ingest_stats['embeddings_stored']:>6}")
    print(f"  Execution Time           : {duration:>6.3f} seconds")
    if ingest_stats['embeddings_stored'] > 0:
        print(f"  Throughput               : {ingest_stats['embeddings_stored'] / duration:>6.1f} chunks/sec")
    print(f"  Total Model Embeddings   : {counts['model_embeddings']:>6}")
    print(f"  Coverage Percentage      : {counts['coverage_pct']:>6.1f}%")
    print(f"  Orphan Embeddings        : {counts['orphan_embeddings']:>6}")
    print("-" * 78)

    # 6. Optional Smoke Test
    if args.smoke_test or ingest_stats['embeddings_stored'] > 0:
        run_smoke_test(store)

    print("\n" + "=" * 78)
    print("  VECTOR EMBEDDING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
