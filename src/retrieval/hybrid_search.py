"""
Hybrid Relational-Plus-Vector Query Execution Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements Pre-Filtering (Attribute-First), Post-Filtering (Vector-First), and
Joint Execution Strategies for hybrid clinical information retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.retrieval.vector_search import VectorSearchEngine

class HybridSearchEngine:
    """Executes hybrid relational and semantic vector queries."""

    def __init__(self, metric: str = "cosine"):
        self.vector_engine = VectorSearchEngine(metric=metric)

    # -------------------------------------------------------------------------
    # In-Memory Simulation for Standalone Benchmarking & Tests
    # -------------------------------------------------------------------------
    def pre_filter_search(
        self,
        query_vector: np.ndarray,
        corpus: List[Dict[str, Any]],
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Pre-filtering (Attribute-First):
        1. Evaluates relational filters over structured metadata first.
        2. Computes vector distances ONLY on surviving rows.
        """
        # Step 1: Relational filter
        surviving_rows = []
        for row in corpus:
            match = True
            for k, expected_v in filters.items():
                if row.get(k) != expected_v:
                    match = False
                    break
            if match:
                surviving_rows.append(row)

        if not surviving_rows:
            return []

        # Step 2: Compute vector distances on filtered subset
        results = []
        for row in surviving_rows:
            vec = np.array(row["embedding"], dtype=np.float32)
            dist = self.vector_engine.compute_distance(query_vector, vec)
            results.append({**row, "distance": dist})

        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def post_filter_search(
        self,
        query_vector: np.ndarray,
        corpus: List[Dict[str, Any]],
        filters: Dict[str, Any],
        top_k: int = 10,
        oversample_factor: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Post-filtering (Vector-First):
        1. Retrieves top-M candidates globally based on vector similarity (M = top_k * oversample_factor).
        2. Applies relational filters to prune candidates down to top-K.
        """
        # Step 1: Global vector distance
        scored = []
        for row in corpus:
            vec = np.array(row["embedding"], dtype=np.float32)
            dist = self.vector_engine.compute_distance(query_vector, vec)
            scored.append((row, dist))

        scored.sort(key=lambda x: x[1])
        top_m = scored[:top_k * oversample_factor]

        # Step 2: Prune by relational filters
        results = []
        for row, dist in top_m:
            match = True
            for k, expected_v in filters.items():
                if row.get(k) != expected_v:
                    match = False
                    break
            if match:
                results.append({**row, "distance": dist})
                if len(results) >= top_k:
                    break

        return results

    # -------------------------------------------------------------------------
    # PostgreSQL pgvector Hybrid SQL Query Builders
    # -------------------------------------------------------------------------
    def build_sql_pre_filtered(
        self,
        query_vector: List[float],
        department: str,
        encounter_type: str = "inpatient",
        top_k: int = 10
    ) -> str:
        """
        Attribute-first hybrid query.
        Forces relational filtering before vector ranking using CTE or subquery.
        """
        vec_literal = self.vector_engine.to_pgvector_literal(query_vector)
        return f"""
        -- Hybrid Strategy 1: Attribute-First (Pre-Filtering)
        WITH filtered_candidates AS (
            SELECT 
                ce.embedding_id,
                ce.chunk_id,
                ce.embedding,
                dc.chunk_text,
                cd.title AS doc_title,
                cd.document_type,
                e.department,
                e.encounter_type,
                e.admission_date,
                p.synthetic_mrn
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
            JOIN clinical_documents cd ON dc.document_id = cd.document_id
            JOIN encounters e ON cd.encounter_id = e.encounter_id
            JOIN patients p ON e.patient_id = p.patient_id
            WHERE e.department = '{department}'
              AND e.encounter_type = '{encounter_type}'
        )
        SELECT 
            embedding_id,
            chunk_id,
            chunk_text,
            doc_title,
            department,
            synthetic_mrn,
            (embedding <=> '{vec_literal}'::vector) AS cosine_distance
        FROM filtered_candidates
        ORDER BY embedding <=> '{vec_literal}'::vector
        LIMIT {top_k};
        """

    def build_sql_post_filtered(
        self,
        query_vector: List[float],
        department: str,
        encounter_type: str = "inpatient",
        top_k: int = 10,
        oversample_m: int = 50
    ) -> str:
        """
        Vector-first hybrid query (Post-Filtering).
        Retrieves top-M nearest vectors first, then filters by relational attributes.
        """
        vec_literal = self.vector_engine.to_pgvector_literal(query_vector)
        return f"""
        -- Hybrid Strategy 2: Vector-First (Post-Filtering)
        WITH nearest_vectors AS (
            SELECT 
                ce.embedding_id,
                ce.chunk_id,
                (ce.embedding <=> '{vec_literal}'::vector) AS cosine_distance
            FROM chunk_embeddings ce
            ORDER BY ce.embedding <=> '{vec_literal}'::vector
            LIMIT {oversample_m}
        )
        SELECT 
            nv.embedding_id,
            dc.chunk_id,
            dc.chunk_text,
            cd.title AS doc_title,
            e.department,
            e.encounter_type,
            p.synthetic_mrn,
            nv.cosine_distance
        FROM nearest_vectors nv
        JOIN document_chunks dc ON nv.chunk_id = dc.chunk_id
        JOIN clinical_documents cd ON dc.document_id = cd.document_id
        JOIN encounters e ON cd.encounter_id = e.encounter_id
        JOIN patients p ON e.patient_id = p.patient_id
        WHERE e.department = '{department}'
          AND e.encounter_type = '{encounter_type}'
        ORDER BY nv.cosine_distance
        LIMIT {top_k};
        """

    def build_sql_joint_hybrid(
        self,
        query_vector: List[float],
        department: str,
        icd10_code: Optional[str] = None,
        top_k: int = 10
    ) -> str:
        """
        Integrated joint hybrid query allowing the PostgreSQL query optimizer
        to select index scan vs bitmap index scan based on selectivity.
        """
        vec_literal = self.vector_engine.to_pgvector_literal(query_vector)
        icd_clause = f"AND diag.icd10_code = '{icd10_code}'" if icd10_code else ""
        return f"""
        -- Hybrid Strategy 3: Integrated Joint Query Optimization
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT 
            ce.embedding_id,
            dc.chunk_text,
            cd.title,
            e.department,
            p.synthetic_mrn,
            (ce.embedding <=> '{vec_literal}'::vector) AS cosine_distance
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
        JOIN clinical_documents cd ON dc.document_id = cd.document_id
        JOIN encounters e ON cd.encounter_id = e.encounter_id
        JOIN patients p ON e.patient_id = p.patient_id
        LEFT JOIN diagnoses diag ON e.encounter_id = diag.encounter_id
        WHERE e.department = '{department}'
          {icd_clause}
        ORDER BY ce.embedding <=> '{vec_literal}'::vector
        LIMIT {top_k};
        """
