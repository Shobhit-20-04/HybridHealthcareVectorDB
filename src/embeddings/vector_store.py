"""
PostgreSQL pgvector Storage & Vector Management Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Manages:
1. Model metadata verification and registration in embedding_models.
2. PostgreSQL column dimension verification against pg_attribute.
3. Batch vector embedding generation and idempotent pgvector upserts into chunk_embeddings.
4. Referential and dimension invariant verification.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("embeddings.vector_store")

class PgVectorStore:
    """Manages idempotent vector generation and storage in PostgreSQL via pgvector."""

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.embedder = embedder or TextEmbedder(model_name=model_id, dimension=dimension)

    def check_and_register_model(
        self,
        model_name: Optional[str] = None,
        distance_metric: str = "cosine",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verifies that the embedding model is registered in embedding_models table.
        If missing, registers it with dimension and metric metadata.
        """
        clean_name = model_name or f"Model {self.model_id}"
        clean_desc = description or f"Registered dense vector model with {self.dimension} dimensions."

        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                # Check existing registration
                cur.execute(
                    "SELECT model_id, model_name, dimension, distance_metric FROM embedding_models WHERE model_id = %s;",
                    (self.model_id,)
                )
                row = cur.fetchone()
                if row:
                    db_model_id, db_name, db_dim, db_metric = row
                    if db_dim != self.dimension:
                        raise DimensionMismatchError(
                            f"Model '{self.model_id}' is registered with dimension {db_dim}, "
                            f"but current embedder expects {self.dimension}."
                        )
                    logger.info(f"[V] Embedding model verified in registry: {self.model_id} ({db_dim}-d, {db_metric})")
                    return {
                        "model_id": db_model_id,
                        "model_name": db_name,
                        "dimension": db_dim,
                        "distance_metric": db_metric
                    }

                # Register model
                logger.info(f"Registering new model '{self.model_id}' in embedding_models table...")
                cur.execute("""
                    INSERT INTO embedding_models (
                        model_id, model_name, modality, dimension, distance_metric, description
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (model_id) DO UPDATE SET
                        dimension = EXCLUDED.dimension,
                        distance_metric = EXCLUDED.distance_metric;
                """, (self.model_id, clean_name, "text", self.dimension, distance_metric, clean_desc))

        return {
            "model_id": self.model_id,
            "model_name": clean_name,
            "dimension": self.dimension,
            "distance_metric": distance_metric
        }

    def verify_column_dimension(self, table_name: str = "chunk_embeddings", column_name: str = "embedding") -> int:
        """
        Queries PostgreSQL catalog (pg_attribute.atttypmod) to ensure the target column
        strictly matches the model's output vector dimension.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT a.atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE c.relname = %s 
                      AND a.attname = %s
                      AND a.attnum > 0 
                      AND NOT a.attisdropped;
                """, (table_name, column_name))
                row = cur.fetchone()

                if not row or row[0] is None:
                    raise DimensionMismatchError(
                        f"Could not resolve pgvector dimension for column '{column_name}' on table '{table_name}'."
                    )

                column_dim = row[0]
                if column_dim != self.dimension:
                    raise DimensionMismatchError(
                        f"Database column '{table_name}.{column_name}' expects vector({column_dim}), "
                        f"but model '{self.model_id}' outputs {self.dimension} dimensions. Ingestion halted."
                    )

                logger.info(f"[V] Database column '{table_name}.{column_name}' dimension validated: {column_dim} dims.")
                return column_dim

    def ingest_chunk_embeddings(
        self,
        batch_size: int = 32,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Generates and stores vector embeddings for document chunks using atomic transactions.
        Idempotent: Uses ON CONFLICT (chunk_id, model_id) DO UPDATE so re-running never duplicates rows.
        """
        # Step 1: Pre-flight verifications
        self.check_and_register_model()
        self.verify_column_dimension("chunk_embeddings", "embedding")

        # Step 2: Fetch target chunks
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                if force:
                    cur.execute("""
                        SELECT chunk_id, document_id, chunk_index, chunk_text
                        FROM document_chunks
                        ORDER BY document_id, chunk_index;
                    """)
                else:
                    cur.execute("""
                        SELECT dc.chunk_id, dc.document_id, dc.chunk_index, dc.chunk_text
                        FROM document_chunks dc
                        LEFT JOIN chunk_embeddings ce 
                          ON dc.chunk_id = ce.chunk_id AND ce.model_id = %s
                        WHERE ce.embedding_id IS NULL
                        ORDER BY dc.document_id, dc.chunk_index;
                    """, (self.model_id,))
                
                chunks_to_process = cur.fetchall()

        total_chunks = len(chunks_to_process)
        logger.info(f"Identified {total_chunks} chunks to vectorize with model '{self.model_id}'.")

        if total_chunks == 0:
            logger.info("[V] All document chunks are already vectorized in PostgreSQL. No new work needed.")
            return {
                "chunks_processed": 0,
                "embeddings_stored": 0,
                "model_id": self.model_id
            }

        total_inserted = 0

        # Step 3: Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks_to_process[i:i + batch_size]
            texts = [b[3] for b in batch]

            # Generate vectors
            vectors = self.embedder.embed_batch(texts, batch_size=batch_size)

            insert_rows = []
            for (chunk_id, doc_id, chunk_idx, _), vec in zip(batch, vectors):
                # Deterministic UUIDv5 ensures idempotent embedding_id
                emb_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk_id}:{self.model_id}"))
                insert_rows.append((
                    emb_id,
                    str(chunk_id),
                    self.model_id,
                    self.dimension,
                    vec
                ))

            # Atomic transaction per batch
            with get_connection(autocommit=False) as conn:
                with conn.cursor() as cur:
                    upsert_stmt = """
                        INSERT INTO chunk_embeddings (
                            embedding_id, chunk_id, model_id, dimension, embedding, created_at
                        ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (chunk_id, model_id) DO UPDATE SET
                            dimension = EXCLUDED.dimension,
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP;
                    """
                    cur.executemany(upsert_stmt, insert_rows)
                conn.commit()

            total_inserted += len(insert_rows)
            logger.info(f"  -> Vectorized and stored batch {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}")

        logger.info(f"[V] Vector storage completed: {total_inserted} chunk embeddings successfully saved in pgvector.")
        return {
            "chunks_processed": total_chunks,
            "embeddings_stored": total_inserted,
            "model_id": self.model_id
        }

    def get_embedding_counts(self) -> Dict[str, Any]:
        """Queries table statistics and reconciles documents, chunks, and embeddings."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                total_docs = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM document_chunks;")
                total_chunks = cur.fetchone()[0]

                cur.execute(
                    "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = %s;",
                    (self.model_id,)
                )
                model_embeddings = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                total_embeddings = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*) 
                    FROM chunk_embeddings ce
                    LEFT JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    WHERE dc.chunk_id IS NULL;
                """)
                orphans = cur.fetchone()[0]

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "model_embeddings": model_embeddings,
            "total_embeddings": total_embeddings,
            "unembedded_chunks": max(0, total_chunks - model_embeddings),
            "orphan_embeddings": orphans,
            "coverage_pct": round((model_embeddings / total_chunks * 100), 2) if total_chunks > 0 else 0.0
        }
