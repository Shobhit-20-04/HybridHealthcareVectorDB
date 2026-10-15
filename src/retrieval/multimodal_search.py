"""
Multimodal Healthcare Retrieval Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. Image-to-Image Similarity Retrieval:
   - Takes a query radiological image (or pre-computed 512-d vision vector)
   - Performs cosine distance (<=>) search over image_embeddings in PostgreSQL
   - Relational joins across medical_images, encounters, patients, and clinical_documents
2. Cross-Modal Text-to-Image Retrieval:
   - Encodes clinical natural language queries into 512-d latent space using aligned CLIP text-tower
   - Queries image_embeddings to discover relevant diagnostic imaging studies
3. Strict Embedding Space Protection:
   - Rejects cross-space queries (e.g., passing 384-d text embeddings into 512-d image space)
   - Enforces catalog-level type invariants
4. Execution Plan Inspection:
   - Supports EXPLAIN (ANALYZE, BUFFERS) for query plan and index verification
"""

import time
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from database.connection import get_connection
from src.embeddings.image_store import IncompatibleEmbeddingSpaceError
from src.embeddings.image_embedder import ImageEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retrieval.multimodal_search")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class MultimodalSearchEngine:
    """
    Executes multimodal image-to-image and cross-modal text-to-image queries
    against PostgreSQL with pgvector, combining relational filters with vector ranking.
    """

    DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"
    DEFAULT_DIMENSION = 512

    def __init__(
        self,
        model_id: str = DEFAULT_IMAGE_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        embedder: Optional[ImageEmbedder] = None
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.embedder = embedder or ImageEmbedder(model_name=model_id, dimension=dimension)

    def validate_vector_space(self, vector: List[float]) -> None:
        """
        Guards against vector dimension pollution.
        Strictly rejects 384-d or non-512-d vectors from querying multimodal space.
        """
        if not isinstance(vector, (list, tuple)):
            raise IncompatibleEmbeddingSpaceError(f"Vector must be a list/tuple of floats, got {type(vector)}")
        if len(vector) != self.dimension:
            raise IncompatibleEmbeddingSpaceError(
                f"Incompatible embedding space: cannot compare {len(vector)}-dimensional vector "
                f"with {self.dimension}-dimensional multimodal image space ('{self.model_id}')."
            )

    def build_parameterized_query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        modality: Optional[Union[str, List[str]]] = None,
        body_site: Optional[Union[str, List[str]]] = None,
        department: Optional[Union[str, List[str]]] = None,
        encounter_type: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None
    ) -> Tuple[str, Tuple]:
        """
        Compiles a parameterized SQL query joining image_embeddings, medical_images,
        encounters, patients, and clinical documents.
        """
        self.validate_vector_space(query_vector)

        where_clauses = ["ie.model_id = %s"]
        params: List[Any] = [self.model_id]

        if modality:
            if isinstance(modality, (list, tuple, set)):
                where_clauses.append("mi.modality = ANY(%s)")
                params.append(list(modality))
            else:
                where_clauses.append("mi.modality = %s")
                params.append(modality)

        if body_site:
            if isinstance(body_site, (list, tuple, set)):
                where_clauses.append("mi.body_site = ANY(%s)")
                params.append(list(body_site))
            else:
                where_clauses.append("mi.body_site = %s")
                params.append(body_site)

        if department:
            if isinstance(department, (list, tuple, set)):
                where_clauses.append("e.department = ANY(%s)")
                params.append(list(department))
            else:
                where_clauses.append("e.department = %s")
                params.append(department)

        if encounter_type:
            where_clauses.append("e.encounter_type = %s")
            params.append(encounter_type)

        if min_age is not None:
            where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) >= %s")
            params.append(int(min_age))

        if max_age is not None:
            where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) <= %s")
            params.append(int(max_age))

        # Vector parameter for projection and ordering
        # Need query_vector in SELECT (distance, similarity) and ORDER BY, plus top_k in LIMIT
        full_params = [query_vector, query_vector] + params + [query_vector, int(top_k)]
        predicates_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT 
                ie.image_id,
                mi.synthetic_image_num,
                mi.modality,
                mi.body_site,
                mi.image_format,
                mi.synthetic_file_path,
                mi.file_checksum_sha256,
                mi.file_size_bytes,
                mi.is_verified_asset,
                p.patient_id,
                p.synthetic_mrn,
                p.gender,
                EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth))::int AS patient_age,
                e.encounter_id,
                e.department,
                e.encounter_type,
                e.admission_date,
                cd.document_id,
                cd.title AS associated_report_title,
                (ie.embedding <=> %s::vector) AS cosine_distance,
                (1.0 - (ie.embedding <=> %s::vector)) AS cosine_similarity
            FROM image_embeddings ie
            JOIN medical_images mi ON ie.image_id = mi.image_id
            JOIN encounters e ON mi.encounter_id = e.encounter_id
            JOIN patients p ON mi.patient_id = p.patient_id
            LEFT JOIN clinical_documents cd ON mi.document_id = cd.document_id
            WHERE {predicates_sql}
            ORDER BY ie.embedding <=> %s::vector ASC
            LIMIT %s;
        """

        return sql, tuple(full_params)

    @staticmethod
    def _parse_plan_summary(plan_lines: List[str]) -> Dict[str, Any]:
        """Parses execution plan to verify index utilization."""
        used_indexes: List[str] = []
        has_seq_scan = False

        for line in plan_lines:
            if "Index Scan using " in line:
                idx = line.split("Index Scan using ")[1].split(" ")[0]
                used_indexes.append(idx)
            elif "Seq Scan on " in line:
                has_seq_scan = True

        strategy = "index_scan" if used_indexes else ("sequential_scan" if has_seq_scan else "nested_scan")

        return {
            "strategy": strategy,
            "indexes_scanned": used_indexes,
            "has_seq_scan": has_seq_scan
        }

    def _execute_query(
        self,
        sql: str,
        params: Tuple,
        explain: bool = False
    ) -> Tuple[List[Dict[str, Any]], float, List[str], Dict[str, Any]]:
        """Executes the compiled SQL query with optional EXPLAIN analysis."""
        plan_lines: List[str] = []
        plan_summary: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        t_start = time.perf_counter()
        with get_connection() as conn:
            with conn.cursor() as cur:
                if explain:
                    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"
                    cur.execute(explain_sql, params)
                    plan_lines = [row[0] for row in cur.fetchall()]
                    plan_summary = self._parse_plan_summary(plan_lines)

                cur.execute(sql, params)
                rows = cur.fetchall()

                for rank, r in enumerate(rows, start=1):
                    results.append({
                        "rank": rank,
                        "image_id": str(r[0]),
                        "synthetic_image_num": r[1],
                        "modality": r[2],
                        "body_site": r[3],
                        "image_format": r[4],
                        "synthetic_file_path": r[5],
                        "file_checksum_sha256": r[6],
                        "file_size_bytes": r[7],
                        "is_verified_asset": bool(r[8]),
                        "patient_id": str(r[9]),
                        "synthetic_mrn": r[10],
                        "gender": r[11],
                        "patient_age": r[12],
                        "encounter_id": str(r[13]),
                        "department": r[14],
                        "encounter_type": r[15],
                        "admission_date": str(r[16]),
                        "document_id": str(r[17]) if r[17] else None,
                        "associated_report_title": r[18],
                        "cosine_distance": float(r[19]),
                        "cosine_similarity": float(r[20])
                    })

        t_ms = (time.perf_counter() - t_start) * 1000.0
        return results, t_ms, plan_lines, plan_summary

    def search_image_to_image(
        self,
        query_image: Union[str, Path, List[float]],
        top_k: int = 10,
        modality: Optional[Union[str, List[str]]] = None,
        body_site: Optional[Union[str, List[str]]] = None,
        department: Optional[Union[str, List[str]]] = None,
        encounter_type: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Image-to-Image visual similarity search:
        Given a query image file or precomputed 512-d vision vector,
        retrieves nearest imaging studies ranked by cosine distance (<=>).
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        t_embed_start = time.perf_counter()
        if isinstance(query_image, (list, tuple)):
            query_vector = list(query_image)
            self.validate_vector_space(query_vector)
            source_desc = f"custom_vector_{len(query_vector)}d"
        else:
            img_path = PROJECT_ROOT / query_image if not Path(query_image).is_absolute() else Path(query_image)
            if not img_path.exists():
                raise FileNotFoundError(f"Query image file does not exist: {img_path}")
            query_vector = self.embedder.embed_image_path(img_path)
            self.validate_vector_space(query_vector)
            source_desc = str(img_path.name)

        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        sql, params = self.build_parameterized_query(
            query_vector=query_vector,
            top_k=top_k,
            modality=modality,
            body_site=body_site,
            department=department,
            encounter_type=encounter_type,
            min_age=min_age,
            max_age=max_age
        )

        results, t_db_ms, plan_lines, plan_summary = self._execute_query(sql, params, explain=explain)

        return {
            "search_type": "image_to_image",
            "query_source": source_desc,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "top_k": top_k,
            "returned_count": len(results),
            "filters_applied": {
                "modality": modality,
                "body_site": body_site,
                "department": department,
                "encounter_type": encounter_type,
                "min_age": min_age,
                "max_age": max_age
            },
            "embedding_time_ms": round(t_embed_ms, 3),
            "database_time_ms": round(t_db_ms, 3),
            "total_time_ms": round(t_embed_ms + t_db_ms, 3),
            "plan_summary": plan_summary,
            "query_plan": plan_lines,
            "results": results
        }

    def search_text_to_image(
        self,
        query_text: str,
        top_k: int = 10,
        modality: Optional[Union[str, List[str]]] = None,
        body_site: Optional[Union[str, List[str]]] = None,
        department: Optional[Union[str, List[str]]] = None,
        encounter_type: Optional[str] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Cross-Modal Text-to-Image retrieval:
        Encodes clinical query text via the aligned CLIP text tower into the shared
        512-d latent space and finds matching medical imaging assets.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        t_embed_start = time.perf_counter()
        query_vector = self.embedder.embed_query_text(query_text.strip())
        self.validate_vector_space(query_vector)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

        sql, params = self.build_parameterized_query(
            query_vector=query_vector,
            top_k=top_k,
            modality=modality,
            body_site=body_site,
            department=department,
            encounter_type=encounter_type,
            min_age=min_age,
            max_age=max_age
        )

        results, t_db_ms, plan_lines, plan_summary = self._execute_query(sql, params, explain=explain)

        return {
            "search_type": "cross_modal_text_to_image",
            "query_text": query_text,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "top_k": top_k,
            "returned_count": len(results),
            "filters_applied": {
                "modality": modality,
                "body_site": body_site,
                "department": department,
                "encounter_type": encounter_type,
                "min_age": min_age,
                "max_age": max_age
            },
            "embedding_time_ms": round(t_embed_ms, 3),
            "database_time_ms": round(t_db_ms, 3),
            "total_time_ms": round(t_embed_ms + t_db_ms, 3),
            "plan_summary": plan_summary,
            "query_plan": plan_lines,
            "results": results
        }
