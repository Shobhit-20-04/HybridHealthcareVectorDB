"""
Hybrid Relational-Plus-Vector Query Execution Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. PostgresHybridSearchEngine: Full production database execution engine supporting:
   - Parameterized metadata predicates (department, document_type, age ranges, ICD-10 codes)
   - Relational joins across patients, encounters, clinical documents, chunks, and diagnoses
   - Cosine distance ranking (<=>) and cosine similarity projection
   - Search modes: exact_flat, hnsw, ivfflat, planner_choice
   - Execution plan extraction and inspection via EXPLAIN (ANALYZE, BUFFERS)
2. HybridSearchEngine: In-memory simulation and SQL builders for unit test compatibility.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import numpy as np

from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError
from src.retrieval.vector_search import VectorSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retrieval.hybrid_search")

class HybridQueryError(Exception):
    """Base exception for hybrid retrieval errors."""
    pass

class PostgresHybridSearchEngine:
    """
    Executes hybrid relational-plus-vector queries against PostgreSQL with pgvector.
    Combines structured SQL metadata predicates with dense vector similarity ranking.
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

    def build_parameterized_query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        department: Optional[Union[str, List[str]]] = None,
        document_type: Optional[Union[str, List[str]]] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        icd10_code: Optional[Union[str, List[str]]] = None
    ) -> Tuple[str, Tuple]:
        """
        Constructs a safe, fully parameterized SQL query with relational joins and predicates.
        Returns the SQL string and the parameter tuple (never raw string interpolation).
        """
        where_clauses = ["ce.model_id = %s"]
        params: List[Any] = [query_vector, query_vector, self.model_id]

        if department:
            if isinstance(department, (list, tuple, set)):
                where_clauses.append("e.department = ANY(%s)")
                params.append(list(department))
            else:
                where_clauses.append("e.department = %s")
                params.append(department)

        if document_type:
            if isinstance(document_type, (list, tuple, set)):
                where_clauses.append("cd.document_type = ANY(%s)")
                params.append(list(document_type))
            else:
                where_clauses.append("cd.document_type = %s")
                params.append(document_type)

        if min_age is not None:
            where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) >= %s")
            params.append(int(min_age))

        if max_age is not None:
            where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) <= %s")
            params.append(int(max_age))

        if icd10_code:
            if isinstance(icd10_code, (list, tuple, set)):
                where_clauses.append("diag.icd10_code = ANY(%s)")
                params.append(list(icd10_code))
            else:
                where_clauses.append("diag.icd10_code = %s")
                params.append(icd10_code)

        # Order by placeholder and Limit placeholder
        params.append(query_vector)
        params.append(int(top_k))

        predicates_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT 
                ce.chunk_id,
                dc.chunk_index,
                dc.chunk_text,
                dc.char_length,
                dc.token_count,
                cd.document_id,
                cd.synthetic_document_num,
                cd.title AS document_title,
                cd.document_type,
                cd.metadata AS doc_metadata,
                p.patient_id,
                p.synthetic_mrn,
                p.gender,
                p.date_of_birth,
                EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth))::int AS patient_age,
                e.encounter_id,
                e.department,
                e.encounter_type,
                e.admission_date,
                COALESCE(
                    STRING_AGG(DISTINCT diag.icd10_code || ': ' || diag.diagnosis_description, '; '),
                    'None Recorded'
                ) AS diagnoses_summary,
                (ce.embedding <=> %s::vector) AS cosine_distance,
                (1.0 - (ce.embedding <=> %s::vector)) AS cosine_similarity
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
            JOIN clinical_documents cd ON dc.document_id = cd.document_id
            JOIN encounters e ON cd.encounter_id = e.encounter_id
            JOIN patients p ON cd.patient_id = p.patient_id
            LEFT JOIN diagnoses diag ON e.encounter_id = diag.encounter_id
            WHERE {predicates_sql}
            GROUP BY 
                ce.chunk_id, dc.chunk_index, dc.chunk_text, dc.char_length, dc.token_count,
                cd.document_id, cd.synthetic_document_num, cd.title, cd.document_type, cd.metadata,
                p.patient_id, p.synthetic_mrn, p.gender, p.date_of_birth,
                e.encounter_id, e.department, e.encounter_type, e.admission_date, ce.embedding
            ORDER BY ce.embedding <=> %s::vector ASC
            LIMIT %s;
        """

        return sql, tuple(params)

    @staticmethod
    def _parse_plan_summary(plan_lines: List[str]) -> Dict[str, Any]:
        """
        Parses EXPLAIN (ANALYZE, BUFFERS) text to determine optimizer node execution paths.
        """
        plan_text = " ".join(plan_lines)
        used_indexes: List[str] = []
        used_bitmap_indexes: List[str] = []

        for line in plan_lines:
            if "Index Scan using " in line:
                idx = line.split("Index Scan using ")[1].split(" ")[0]
                used_indexes.append(idx)
            elif "Bitmap Index Scan on " in line:
                bidx = line.split("Bitmap Index Scan on ")[1].split(" ")[0]
                used_bitmap_indexes.append(bidx)

        has_seq_scan = "Seq Scan on " in plan_text

        # Determine strategy chosen by query planner
        if used_bitmap_indexes and any("idx_clinical_docs" in b or "idx_encounters" in b for b in used_bitmap_indexes):
            strategy = "pre_filter_relational_scan"
        elif any("idx_chunk_embeddings" in i for i in used_indexes):
            strategy = "iterative_vector_scan"
        elif has_seq_scan and not used_indexes:
            strategy = "exact_sequential_scan"
        else:
            strategy = "hybrid_nested_scan"

        return {
            "strategy": strategy,
            "indexes_scanned": used_indexes,
            "bitmap_indexes_scanned": used_bitmap_indexes,
            "has_seq_scan": has_seq_scan
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        department: Optional[Union[str, List[str]]] = None,
        document_type: Optional[Union[str, List[str]]] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        icd10_code: Optional[Union[str, List[str]]] = None,
        search_mode: str = "exact_flat",
        session_params: Optional[Dict[str, Any]] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Executes hybrid relational-plus-vector query with complete metadata joins.
        """
        # 1. Input Validation
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if min_age is not None and max_age is not None and min_age > max_age:
            raise ValueError(f"min_age ({min_age}) cannot be greater than max_age ({max_age})")

        # 2. Vector Encoding
        t_embed_start = time.perf_counter()
        query_vector = self.embedder.embed_text(query.strip())
        self.embedder.validate_dimension(query_vector)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        # 3. Query Compilation
        sql, params = self.build_parameterized_query(
            query_vector=query_vector,
            top_k=top_k,
            department=department,
            document_type=document_type,
            min_age=min_age,
            max_age=max_age,
            icd10_code=icd10_code
        )

        # 4. Session Knobs Configuration
        knobs: Dict[str, Any] = {}
        if search_mode == "exact_flat":
            knobs["enable_indexscan"] = "off"
            knobs["enable_bitmapscan"] = "off"
            knobs["enable_seqscan"] = "on"
        elif search_mode == "hnsw":
            knobs["enable_indexscan"] = "on"
            knobs["enable_seqscan"] = "off"
            knobs["hnsw.ef_search"] = 40
        elif search_mode == "ivfflat":
            knobs["enable_indexscan"] = "on"
            knobs["enable_seqscan"] = "off"
            knobs["ivfflat.probes"] = 5
        elif search_mode in ("planner_choice", "joint"):
            # Allow PostgreSQL cost optimizer full freedom
            pass
        else:
            raise ValueError(f"Unknown search_mode '{search_mode}'. Choose from: exact_flat, hnsw, ivfflat, planner_choice")

        if session_params:
            knobs.update(session_params)

        # 5. Database Execution
        plan_lines: List[str] = []
        plan_summary: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        t_db_start = time.perf_counter()
        with get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Apply session configuration
                    for k, v in knobs.items():
                        cur.execute(f"SET LOCAL {k} = {v};")

                    # Optional EXPLAIN ANALYZE
                    if explain:
                        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"
                        cur.execute(explain_sql, params)
                        plan_lines = [row[0] for row in cur.fetchall()]
                        plan_summary = self._parse_plan_summary(plan_lines)

                    # Execute query
                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    for rank, r in enumerate(rows, start=1):
                        results.append({
                            "rank": rank,
                            "chunk_id": str(r[0]),
                            "chunk_index": r[1],
                            "snippet": r[2],
                            "char_length": r[3],
                            "token_count": r[4],
                            "document_id": str(r[5]),
                            "synthetic_document_num": r[6],
                            "document_title": r[7],
                            "document_type": r[8],
                            "doc_metadata": r[9],
                            "patient_id": str(r[10]),
                            "synthetic_mrn": r[11],
                            "gender": r[12],
                            "date_of_birth": str(r[13]),
                            "patient_age": r[14],
                            "encounter_id": str(r[15]),
                            "department": r[16],
                            "encounter_type": r[17],
                            "admission_date": str(r[18]),
                            "diagnoses": r[19],
                            "cosine_distance": float(r[20]),
                            "cosine_similarity": float(r[21])
                        })

        t_db_ms = (time.perf_counter() - t_db_start) * 1000.0

        return {
            "query": query,
            "search_mode": search_mode,
            "model_id": self.model_id,
            "dimension": self.expected_dimension,
            "top_k": top_k,
            "returned_count": len(results),
            "filters_applied": {
                "department": department,
                "document_type": document_type,
                "min_age": min_age,
                "max_age": max_age,
                "icd10_code": icd10_code
            },
            "embedding_time_ms": round(t_embed_ms, 3),
            "database_time_ms": round(t_db_ms, 3),
            "total_time_ms": round(t_embed_ms + t_db_ms, 3),
            "plan_summary": plan_summary,
            "query_plan": plan_lines,
            "results": results
        }


# =============================================================================
# In-Memory Simulation & SQL Builders (Maintained for Test Compatibility)
# =============================================================================

class HybridSearchEngine:
    """Executes hybrid relational and semantic vector queries."""

    def __init__(self, metric: str = "cosine"):
        self.vector_engine = VectorSearchEngine(metric=metric)

    def pre_filter_search(
        self,
        query_vector: np.ndarray,
        corpus: List[Dict[str, Any]],
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Pre-filtering (Attribute-First) in-memory simulation."""
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
        """Post-filtering (Vector-First) in-memory simulation."""
        scored = []
        for row in corpus:
            vec = np.array(row["embedding"], dtype=np.float32)
            dist = self.vector_engine.compute_distance(query_vector, vec)
            scored.append((row, dist))

        scored.sort(key=lambda x: x[1])
        top_m = scored[:top_k * oversample_factor]

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

    def build_sql_pre_filtered(
        self,
        query_vector: List[float],
        department: str,
        encounter_type: str = "inpatient",
        top_k: int = 10
    ) -> str:
        """Attribute-first hybrid query."""
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
        """Vector-first hybrid query (Post-Filtering)."""
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
        """Integrated joint hybrid query."""
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
