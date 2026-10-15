"""
Exact Semantic Retrieval Engine (Ground-Truth Flat Search Baseline)
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements exact brute-force cosine distance search across clinical document embeddings
without approximate vector indexes (HNSW/IVFFlat). Serves as the authoritative ground-truth
baseline for ranking accuracy, Recall@K evaluation, and index performance benchmarking.

Mathematical Formulations:
  - Cosine Distance:   D_C(u, v) = 1.0 - (u . v) / (||u||_2 * ||v||_2)  in [0, 2]
  - Cosine Similarity: S_C(u, v) = 1.0 - D_C(u, v)                       in [-1, 1]
  Note: In pgvector, the <=> operator computes Cosine Distance (lower is closer).
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retrieval.exact_search")

class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass

class NoEmbeddingsFoundError(RetrievalError):
    """Raised when no embeddings exist in the database for the requested model."""
    pass

class ExactSemanticSearchEngine:
    """
    Executes exact flat vector similarity search against PostgreSQL with pgvector.
    Forces sequential scan execution (disabling index scans) to guarantee 100% recall ground truth.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        expected_dimension: int = DEFAULT_DIMENSION,
        embedder: Optional[TextEmbedder] = None
    ):
        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.embedder = embedder or TextEmbedder(model_name=model_id, dimension=expected_dimension)

    def verify_compatibility(self) -> Dict[str, Any]:
        """
        Validates model registration, vector column dimension, and stored vector availability.
        Raises DimensionMismatchError or NoEmbeddingsFoundError if incompatible.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Check embedding_models registry
                cur.execute(
                    "SELECT model_id, dimension, distance_metric FROM embedding_models WHERE model_id = %s;",
                    (self.model_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise RetrievalError(f"Model '{self.model_id}' is not registered in embedding_models table.")
                
                db_model_id, db_dim, db_metric = row
                if db_dim != self.expected_dimension:
                    raise DimensionMismatchError(
                        f"Model dimension mismatch: registry specifies {db_dim}, expected {self.expected_dimension}."
                    )

                # 2. Check column dimension in catalog
                cur.execute("""
                    SELECT a.atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    WHERE c.relname = 'chunk_embeddings' AND a.attname = 'embedding' AND a.attnum > 0;
                """)
                col_row = cur.fetchone()
                if not col_row or col_row[0] != self.expected_dimension:
                    actual_dim = col_row[0] if col_row else "unknown"
                    raise DimensionMismatchError(
                        f"Database column 'chunk_embeddings.embedding' expects vector({actual_dim}), "
                        f"but search engine configured for {self.expected_dimension}."
                    )

                # 3. Check stored embeddings count
                cur.execute(
                    "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = %s;",
                    (self.model_id,)
                )
                total_stored = cur.fetchone()[0]
                if total_stored == 0:
                    raise NoEmbeddingsFoundError(
                        f"Zero embeddings found in 'chunk_embeddings' for model '{self.model_id}'. "
                        f"Run 'python scripts/embed_documents.py' first."
                    )

        return {
            "model_id": self.model_id,
            "dimension": self.expected_dimension,
            "distance_metric": db_metric,
            "stored_embeddings_count": total_stored
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Executes exact sequential flat vector search for a natural-language query.
        
        Args:
            query: Natural language query string (cannot be empty).
            top_k: Number of nearest documents to return (must be > 0).
            explain: If True, retrieves and attaches PostgreSQL EXPLAIN ANALYZE plan.
            
        Returns:
            Dictionary containing query metadata, timing, and ranked search hits.
        """
        # 1. Input Validation
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer greater than 0, got {top_k}.")

        # 2. Pre-flight Compatibility Verification
        compat = self.verify_compatibility()

        # 3. Query Vectorization
        t_embed_start = time.perf_counter()
        query_vector = self.embedder.embed_text(query.strip())
        self.embedder.validate_dimension(query_vector)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        query_plan = []
        results = []

        # 4. Database Query with Explicit Sequential Scan Enforcement
        t_query_start = time.perf_counter()
        with get_connection(autocommit=False) as conn:
            with conn.cursor() as cur:
                # Force exact sequential scan by disabling approximate index paths
                cur.execute("SET LOCAL enable_indexscan = off;")
                cur.execute("SET LOCAL enable_bitmapscan = off;")

                # Optional EXPLAIN ANALYZE
                if explain:
                    explain_sql = """
                        EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
                        SELECT 
                            ce.chunk_id,
                            (ce.embedding <=> %s::vector) AS cosine_distance
                        FROM chunk_embeddings ce
                        WHERE ce.model_id = %s
                        ORDER BY ce.embedding <=> %s::vector ASC
                        LIMIT %s;
                    """
                    cur.execute(explain_sql, (query_vector, self.model_id, query_vector, top_k))
                    query_plan = [row[0] for row in cur.fetchall()]

                # Core Exact Search Query joining clinical entities
                search_sql = """
                    SELECT 
                        ce.embedding_id,
                        ce.chunk_id,
                        dc.document_id,
                        cd.synthetic_document_num,
                        p.synthetic_mrn,
                        cd.patient_id,
                        cd.encounter_id,
                        e.department,
                        cd.document_type,
                        cd.title,
                        dc.chunk_index,
                        dc.token_count,
                        dc.char_length,
                        dc.chunk_text,
                        cd.metadata AS doc_metadata,
                        (ce.embedding <=> %s::vector) AS cosine_distance,
                        (1.0 - (ce.embedding <=> %s::vector)) AS cosine_similarity
                    FROM chunk_embeddings ce
                    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    JOIN clinical_documents cd ON dc.document_id = cd.document_id
                    JOIN encounters e ON cd.encounter_id = e.encounter_id
                    JOIN patients p ON cd.patient_id = p.patient_id
                    WHERE ce.model_id = %s
                    ORDER BY ce.embedding <=> %s::vector ASC
                    LIMIT %s;
                """
                cur.execute(
                    search_sql,
                    (query_vector, query_vector, self.model_id, query_vector, top_k)
                )

                for rank, row in enumerate(cur.fetchall(), start=1):
                    dist = float(row[15])
                    sim = float(row[16])
                    results.append({
                        "rank": rank,
                        "embedding_id": str(row[0]),
                        "chunk_id": str(row[1]),
                        "document_id": str(row[2]),
                        "synthetic_document_num": row[3],
                        "synthetic_mrn": row[4],
                        "patient_id": str(row[5]),
                        "encounter_id": str(row[6]),
                        "department": row[7],
                        "document_type": row[8],
                        "title": row[9],
                        "chunk_index": row[10],
                        "token_count": row[11],
                        "char_length": row[12],
                        "snippet": row[13],
                        "doc_metadata": row[14],
                        "cosine_distance": dist,
                        "cosine_similarity": sim
                    })

        t_query_ms = (time.perf_counter() - t_query_start) * 1000.0

        return {
            "query": query,
            "model_id": self.model_id,
            "dimension": self.expected_dimension,
            "top_k": top_k,
            "total_corpus_embeddings": compat["stored_embeddings_count"],
            "returned_count": len(results),
            "search_type": "exact_sequential_scan_flat",
            "embedding_time_ms": round(t_embed_ms, 3),
            "database_time_ms": round(t_query_ms, 3),
            "total_time_ms": round(t_embed_ms + t_query_ms, 3),
            "query_plan": query_plan,
            "results": results
        }
