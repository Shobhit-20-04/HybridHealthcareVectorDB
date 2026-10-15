"""
Controlled Vector Index Experimentation & Benchmarking Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Conducts rigorous empirical comparisons across PostgreSQL with pgvector index structures:
  1. Exact Flat Search (Ground-truth sequential scan, zero indexing overhead, 100% recall)
  2. HNSW (Hierarchical Navigable Small World graph, tuneable m, ef_construction, and ef_search)
  3. IVFFlat (Inverted File Flat Voronoi centroids, tuneable lists and probes, requires trained data)

Key Evaluated Dimensions:
  - Index Build Time (wall-clock milliseconds)
  - Index Disk Footprint (pg_relation_size bytes and human-readable formatting)
  - Query Latency Distribution (Mean, Min, Max, StdDev, P50, P95, P99 milliseconds)
  - Throughput (Queries Per Second / QPS)
  - Accuracy (True Recall@K relative to Exact Search ground truth)
  - Physical Plan Verification via EXPLAIN (ANALYZE, BUFFERS)
"""

import time
import math
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark.vector_index")

# Standardized clinical query test suite covering diverse medical domains
DEFAULT_CLINICAL_QUERIES = [
    "acute myocardial infarction and coronary artery disease with chest pain",
    "type 2 diabetes mellitus with diabetic nephropathy and insulin resistance",
    "congestive heart failure with dyspnea on exertion and pulmonary edema",
    "chronic obstructive pulmonary disease with acute exacerbation and hypoxemia",
    "bacterial pneumonia treated with broad spectrum intravenous antibiotics",
    "ischemic stroke with left-sided hemiparesis and neurological deficit",
    "stage 3 chronic kidney disease with elevated serum creatinine and proteinuria",
    "essential hypertension managed with amlodipine and lisinopril therapy",
    "surgical resection of malignant colorectal adenocarcinoma with lymph node dissection",
    "septic shock secondary to urinary tract infection with severe hypotension",
    "deep vein thrombosis and acute pulmonary embolism on therapeutic anticoagulation",
    "upper gastrointestinal hemorrhage secondary to bleeding peptic ulcer disease",
    "major depressive disorder refractory to selective serotonin reuptake inhibitors",
    "rheumatoid arthritis with bilateral hand joint deformity and elevated CRP",
    "acute severe asthma exacerbation triggered by viral respiratory tract infection"
]

class InsufficientDataForIndexError(ValueError):
    """Raised when attempting to build IVFFlat index on insufficient training vectors."""
    pass

class IndexPlanVerificationError(RuntimeError):
    """Raised when the database execution plan does not match the claimed index path."""
    pass

class VectorIndexExperimentRunner:
    """
    Orchestrates controlled, repeatable vector index benchmarks against PostgreSQL pgvector.
    Maintains strict experimental isolation and guarantees empirical plan verification.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384
    HNSW_INDEX_NAME = "idx_chunk_embeddings_hnsw_cosine"
    IVFFLAT_INDEX_NAME = "idx_chunk_embeddings_ivfflat_cosine"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        expected_dimension: int = DEFAULT_DIMENSION,
        embedder: Optional[TextEmbedder] = None
    ):
        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.embedder = embedder or TextEmbedder(model_name=model_id, dimension=expected_dimension)

    def get_system_metadata(self) -> Dict[str, Any]:
        """
        Retrieves live PostgreSQL version, pgvector extension version, and vector table statistics.
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                # PostgreSQL server version
                cur.execute("SELECT version();")
                pg_version_str = cur.fetchone()[0]

                # pgvector extension version
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                vec_row = cur.fetchone()
                pgvector_version = vec_row[0] if vec_row else "NOT INSTALLED"

                # Stored vectors count for target model
                cur.execute(
                    "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = %s;",
                    (self.model_id,)
                )
                model_vectors_count = cur.fetchone()[0]

                # Total chunk embeddings count
                cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                total_vectors_count = cur.fetchone()[0]

                # Table disk size
                cur.execute("SELECT pg_size_pretty(pg_relation_size('chunk_embeddings')), pg_relation_size('chunk_embeddings');")
                tbl_size_pretty, tbl_size_bytes = cur.fetchone()

        return {
            "postgres_version": pg_version_str,
            "pgvector_version": pgvector_version,
            "model_id": self.model_id,
            "dimension": self.expected_dimension,
            "model_vectors_count": model_vectors_count,
            "total_vectors_count": total_vectors_count,
            "table_disk_size_pretty": tbl_size_pretty,
            "table_disk_size_bytes": tbl_size_bytes
        }

    def drop_vector_indexes(self) -> None:
        """
        Safely removes all vector ANN indexes on chunk_embeddings to establish clean baseline.
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {self.HNSW_INDEX_NAME};")
                cur.execute(f"DROP INDEX IF EXISTS {self.IVFFLAT_INDEX_NAME};")
        logger.info("[V] Dropped existing vector indexes on chunk_embeddings.")

    def create_hnsw_index(
        self,
        m: int = 16,
        ef_construction: int = 64
    ) -> Dict[str, Any]:
        """
        Builds HNSW index with specified graph parameters, timing the creation and measuring disk footprint.
        """
        self.drop_vector_indexes()

        logger.info(f"Building HNSW index ({self.HNSW_INDEX_NAME}) with m={m}, ef_construction={ef_construction}...")
        t_start = time.perf_counter()

        ddl = f"""
            CREATE INDEX {self.HNSW_INDEX_NAME}
            ON chunk_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)

        build_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Measure index size
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        pg_size_pretty(pg_relation_size('{self.HNSW_INDEX_NAME}')),
                        pg_relation_size('{self.HNSW_INDEX_NAME}')
                """)
                size_pretty, size_bytes = cur.fetchone()

        logger.info(f"[V] HNSW index created in {build_time_ms:.2f} ms. Size: {size_pretty} ({size_bytes} bytes).")
        return {
            "index_type": "HNSW",
            "index_name": self.HNSW_INDEX_NAME,
            "m": m,
            "ef_construction": ef_construction,
            "build_time_ms": round(build_time_ms, 3),
            "disk_size_pretty": size_pretty,
            "disk_size_bytes": size_bytes
        }

    def create_ivfflat_index(
        self,
        lists: int = 25
    ) -> Dict[str, Any]:
        """
        Builds IVFFlat index with specified cluster centroid count.
        Enforces strict data sufficiency guard before index creation.
        """
        # Guard: Check that table has sufficient vectors loaded
        meta = self.get_system_metadata()
        stored_count = meta["model_vectors_count"]

        if stored_count == 0:
            raise InsufficientDataForIndexError(
                "Cannot build IVFFlat index: 0 vectors found in 'chunk_embeddings'. "
                "IVFFlat requires existing vectors to compute Voronoi centroids."
            )
        if stored_count < lists:
            raise InsufficientDataForIndexError(
                f"Insufficient vectors to train IVFFlat index: found {stored_count} vectors, "
                f"which is less than requested lists={lists}. Load at least {lists} vectors first."
            )

        self.drop_vector_indexes()

        logger.info(f"Building IVFFlat index ({self.IVFFLAT_INDEX_NAME}) with lists={lists} on {stored_count} vectors...")
        t_start = time.perf_counter()

        ddl = f"""
            CREATE INDEX {self.IVFFLAT_INDEX_NAME}
            ON chunk_embeddings
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)

        build_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Measure index size
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT 
                        pg_size_pretty(pg_relation_size('{self.IVFFLAT_INDEX_NAME}')),
                        pg_relation_size('{self.IVFFLAT_INDEX_NAME}')
                """)
                size_pretty, size_bytes = cur.fetchone()

        logger.info(f"[V] IVFFlat index created in {build_time_ms:.2f} ms. Size: {size_pretty} ({size_bytes} bytes).")
        return {
            "index_type": "IVFFlat",
            "index_name": self.IVFFLAT_INDEX_NAME,
            "lists": lists,
            "build_time_ms": round(build_time_ms, 3),
            "disk_size_pretty": size_pretty,
            "disk_size_bytes": size_bytes
        }

    def restore_default_index(self) -> None:
        """
        Restores the standard production HNSW index configuration after benchmarking completes.
        """
        logger.info("Restoring standard production HNSW vector index...")
        self.create_hnsw_index(m=16, ef_construction=64)

    def verify_execution_plan(
        self,
        expected_index_name: Optional[str],
        session_params: Dict[str, Any],
        sample_vector: List[float],
        top_k: int = 10
    ) -> Tuple[bool, List[str]]:
        """
        Executes EXPLAIN (ANALYZE, BUFFERS) and verifies whether PostgreSQL physically executed
        the expected index scan (or sequential scan for exact baseline).
        """
        with get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Apply session configuration knobs
                    for param, val in session_params.items():
                        cur.execute(f"SET LOCAL {param} = {val};")

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
                    cur.execute(explain_sql, (sample_vector, self.model_id, sample_vector, top_k))
                    plan_lines = [row[0] for row in cur.fetchall()]

        plan_text = " ".join(plan_lines)

        if expected_index_name is None:
            # Exact search baseline: MUST use Seq Scan, MUST NOT use Index Scan
            is_verified = ("Seq Scan on chunk_embeddings" in plan_text) and ("Index Scan" not in plan_text)
        else:
            # Approximate search: MUST use Index Scan using expected_index_name
            expected_clause = f"Index Scan using {expected_index_name} on chunk_embeddings"
            is_verified = expected_clause in plan_text

        return is_verified, plan_lines

    def execute_top_k_query(
        self,
        query_vector: List[float],
        top_k: int,
        session_params: Dict[str, Any]
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Executes a single top-k vector query under transaction-scoped session parameters.
        Returns ordered list of (chunk_id, cosine_distance) and query latency in milliseconds.
        """
        search_sql = """
            SELECT ce.chunk_id, (ce.embedding <=> %s::vector) AS cosine_distance
            FROM chunk_embeddings ce
            WHERE ce.model_id = %s
            ORDER BY ce.embedding <=> %s::vector ASC
            LIMIT %s;
        """
        t_start = time.perf_counter()
        with get_connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    for param, val in session_params.items():
                        cur.execute(f"SET LOCAL {param} = {val};")
                    cur.execute(search_sql, (query_vector, self.model_id, query_vector, top_k))
                    rows = cur.fetchall()
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        retrieved_items = [(str(r[0]), float(r[1])) for r in rows]
        return retrieved_items, latency_ms

    @staticmethod
    def calculate_recalls(
        retrieved_items: List[Tuple[str, float]],
        ground_truth_items: List[Tuple[str, float]],
        k: int,
        epsilon: float = 1e-5
    ) -> Tuple[float, float]:
        """
        Computes both:
          1. Distance-based 1-Recall@K (ANN benchmark standard for ties):
             Fraction of retrieved items whose distance <= D_K + epsilon (where D_K is 
             the K-th ground-truth distance).
          2. Strict ID Recall@K:
             |Retrieved_ANN_IDs[0:K] ∩ Ground_Truth_Exact_IDs[0:K]| / K
        """
        if k <= 0:
            raise ValueError(f"k must be greater than 0, got {k}")

        actual_k = min(k, len(ground_truth_items))
        if actual_k == 0:
            return 1.0, 1.0

        # Strict ID Recall
        retrieved_ids = set(cid for cid, _ in retrieved_items[:k])
        gt_ids = set(cid for cid, _ in ground_truth_items[:k])
        strict_id_recall = len(retrieved_ids.intersection(gt_ids)) / float(actual_k)

        # Distance-based 1-Recall@K
        dk_threshold = ground_truth_items[actual_k - 1][1]
        dist_hits = sum(1 for _, dist in retrieved_items[:k] if dist <= dk_threshold + epsilon)
        distance_recall = dist_hits / float(actual_k)

        return distance_recall, strict_id_recall

    def generate_ground_truth(
        self,
        query_vectors: List[Tuple[str, List[float]]],
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generates ground-truth nearest neighbor rankings using Exact Sequential Flat Search.
        """
        logger.info(f"Generating exact ground truth for {len(query_vectors)} queries (top_k={top_k})...")
        session_params = {
            "enable_seqscan": "on",
            "enable_indexscan": "off",
            "enable_bitmapscan": "off"
        }
        ground_truth = {}
        for query_text, q_vec in query_vectors:
            retrieved_items, _ = self.execute_top_k_query(q_vec, top_k, session_params)
            ground_truth[query_text] = retrieved_items

        return ground_truth

    def evaluate_configuration(
        self,
        config_name: str,
        index_type: str,
        expected_index_name: Optional[str],
        build_metadata: Dict[str, Any],
        session_params: Dict[str, Any],
        query_vectors: List[Tuple[str, List[float]]],
        ground_truth: Dict[str, List[Tuple[str, float]]],
        top_k: int = 10,
        iterations_per_query: int = 3,
        warmup_runs: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluates a single index configuration:
          1. Verifies physical execution plan via EXPLAIN (ANALYZE, BUFFERS).
          2. Runs warmup queries to stabilize cache / buffer pool.
          3. Executes timed iterations across query set.
          4. Computes latency percentiles, throughput (QPS), and Recall@K.
        """
        logger.info(f"Evaluating configuration: {config_name} ({index_type})...")

        # 1. Physical Plan Verification
        sample_vec = query_vectors[0][1]
        is_plan_verified, plan_lines = self.verify_execution_plan(
            expected_index_name=expected_index_name,
            session_params=session_params,
            sample_vector=sample_vec,
            top_k=top_k
        )

        if not is_plan_verified:
            logger.warning(
                f"[!] Execution plan verification FAILED for '{config_name}'. "
                f"Expected index: {expected_index_name}. Check EXPLAIN output."
            )

        # 2. Warmup Runs
        for _ in range(warmup_runs):
            for _, q_vec in query_vectors[:3]:
                self.execute_top_k_query(q_vec, top_k, session_params)

        # 3. Timed Evaluation
        all_latencies_ms: List[float] = []
        distance_recalls: List[float] = []
        strict_id_recalls: List[float] = []

        t_bench_start = time.perf_counter()
        total_query_count = 0

        for query_text, q_vec in query_vectors:
            # Query-level recall (evaluated on first run)
            first_retrieved, first_lat = self.execute_top_k_query(q_vec, top_k, session_params)
            all_latencies_ms.append(first_lat)
            total_query_count += 1

            gt_items = ground_truth.get(query_text, [])
            d_recall, id_recall = self.calculate_recalls(first_retrieved, gt_items, top_k)
            distance_recalls.append(d_recall)
            strict_id_recalls.append(id_recall)

            # Additional timing iterations
            for _ in range(iterations_per_query - 1):
                _, iter_lat = self.execute_top_k_query(q_vec, top_k, session_params)
                all_latencies_ms.append(iter_lat)
                total_query_count += 1

        total_bench_duration_s = time.perf_counter() - t_bench_start
        qps = total_query_count / total_bench_duration_s if total_bench_duration_s > 0 else 0.0

        # 4. Statistical Aggregation
        all_latencies_ms.sort()
        mean_latency = statistics.mean(all_latencies_ms)
        stddev_latency = statistics.stdev(all_latencies_ms) if len(all_latencies_ms) > 1 else 0.0
        min_latency = all_latencies_ms[0]
        max_latency = all_latencies_ms[-1]

        def percentile(data: List[float], p: float) -> float:
            idx = int(math.ceil(p * len(data))) - 1
            return data[max(0, min(idx, len(data) - 1))]

        p50 = percentile(all_latencies_ms, 0.50)
        p95 = percentile(all_latencies_ms, 0.95)
        p99 = percentile(all_latencies_ms, 0.99)
        mean_dist_recall = statistics.mean(distance_recalls) if distance_recalls else 0.0
        mean_strict_id_recall = statistics.mean(strict_id_recalls) if strict_id_recalls else 0.0

        return {
            "config_name": config_name,
            "index_type": index_type,
            "index_name": expected_index_name or "NONE (Sequential Scan)",
            "plan_verified": is_plan_verified,
            "build_time_ms": build_metadata.get("build_time_ms", 0.0),
            "disk_size_pretty": build_metadata.get("disk_size_pretty", "0 bytes"),
            "disk_size_bytes": build_metadata.get("disk_size_bytes", 0),
            "parameters": {
                **build_metadata,
                "session_params": session_params
            },
            "top_k": top_k,
            "total_queries_executed": total_query_count,
            "latency_ms": {
                "mean": round(mean_latency, 3),
                "stddev": round(stddev_latency, 3),
                "min": round(min_latency, 3),
                "max": round(max_latency, 3),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3)
            },
            "throughput_qps": round(qps, 2),
            "recall_at_k": round(mean_dist_recall, 4),
            "strict_id_recall_at_k": round(mean_strict_id_recall, 4),
            "sample_plan": plan_lines[:10]  # First 10 lines of EXPLAIN plan for audit
        }

    def run_controlled_experiment(
        self,
        queries: Optional[List[str]] = None,
        top_k: int = 10,
        iterations_per_query: int = 3,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search_values: Optional[List[int]] = None,
        ivfflat_lists: int = 25,
        ivfflat_probes_values: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Executes the complete end-to-end controlled vector index benchmark suite.
        """
        test_queries = queries or DEFAULT_CLINICAL_QUERIES
        ef_search_grid = hnsw_ef_search_values or [10, 40, 100]
        probes_grid = ivfflat_probes_values or [1, 5, 10]

        logger.info("=" * 78)
        logger.info("  STARTING CONTROLLED VECTOR INDEX EXPERIMENT SUITE")
        logger.info("=" * 78)

        # 1. Retrieve Environment & Catalog Metadata
        sys_meta = self.get_system_metadata()
        logger.info(f"PostgreSQL Version : {sys_meta['postgres_version'].split(' on ')[0]}")
        logger.info(f"pgvector Version   : {sys_meta['pgvector_version']}")
        logger.info(f"Model Registry     : {sys_meta['model_id']} ({sys_meta['dimension']}-dim)")
        logger.info(f"Corpus Embeddings  : {sys_meta['model_vectors_count']} rows")
        logger.info(f"Queries in Suite   : {len(test_queries)} queries (top_k={top_k})")

        # 2. Vectorize all clinical test queries once (shared across all runs)
        logger.info("Vectorizing clinical query suite with Sentence Transformers...")
        query_vectors: List[Tuple[str, List[float]]] = []
        for q_text in test_queries:
            q_vec = self.embedder.embed_text(q_text)
            self.embedder.validate_dimension(q_vec)
            query_vectors.append((q_text, q_vec))

        experiment_results: List[Dict[str, Any]] = []

        try:
            # -----------------------------------------------------------------
            # PHASE 1: Exact Flat Search (Ground-Truth Baseline)
            # -----------------------------------------------------------------
            logger.info("\n>>> PHASE 1: Exact Flat Search (Sequential Scan Baseline)")
            self.drop_vector_indexes()

            ground_truth = self.generate_ground_truth(query_vectors, top_k=top_k)

            flat_session = {
                "enable_seqscan": "on",
                "enable_indexscan": "off",
                "enable_bitmapscan": "off"
            }
            flat_res = self.evaluate_configuration(
                config_name="Exact Flat Search",
                index_type="Flat (Seq Scan)",
                expected_index_name=None,
                build_metadata={"build_time_ms": 0.0, "disk_size_pretty": "0 bytes", "disk_size_bytes": 0},
                session_params=flat_session,
                query_vectors=query_vectors,
                ground_truth=ground_truth,
                top_k=top_k,
                iterations_per_query=iterations_per_query
            )
            experiment_results.append(flat_res)

            # -----------------------------------------------------------------
            # PHASE 2: HNSW Index Parameter Sweeps
            # -----------------------------------------------------------------
            logger.info("\n>>> PHASE 2: HNSW (Hierarchical Navigable Small World)")
            hnsw_build = self.create_hnsw_index(m=hnsw_m, ef_construction=hnsw_ef_construction)

            for ef_s in ef_search_grid:
                hnsw_session = {
                    "enable_seqscan": "off",
                    "enable_indexscan": "on",
                    "hnsw.ef_search": ef_s
                }
                config_title = f"HNSW (m={hnsw_m}, ef_s={ef_s})"
                h_res = self.evaluate_configuration(
                    config_name=config_title,
                    index_type="HNSW",
                    expected_index_name=self.HNSW_INDEX_NAME,
                    build_metadata=hnsw_build,
                    session_params=hnsw_session,
                    query_vectors=query_vectors,
                    ground_truth=ground_truth,
                    top_k=top_k,
                    iterations_per_query=iterations_per_query
                )
                experiment_results.append(h_res)

            # -----------------------------------------------------------------
            # PHASE 3: IVFFlat Index Parameter Sweeps
            # -----------------------------------------------------------------
            logger.info("\n>>> PHASE 3: IVFFlat (Inverted File Flat Centroids)")
            ivfflat_build = self.create_ivfflat_index(lists=ivfflat_lists)

            for prb in probes_grid:
                ivf_session = {
                    "enable_seqscan": "off",
                    "enable_indexscan": "on",
                    "ivfflat.probes": prb
                }
                config_title = f"IVFFlat (lists={ivfflat_lists}, probes={prb})"
                ivf_res = self.evaluate_configuration(
                    config_name=config_title,
                    index_type="IVFFlat",
                    expected_index_name=self.IVFFLAT_INDEX_NAME,
                    build_metadata=ivfflat_build,
                    session_params=ivf_session,
                    query_vectors=query_vectors,
                    ground_truth=ground_truth,
                    top_k=top_k,
                    iterations_per_query=iterations_per_query
                )
                experiment_results.append(ivf_res)

        finally:
            # -----------------------------------------------------------------
            # PHASE 4: Teardown & Production Restoration
            # -----------------------------------------------------------------
            logger.info("\n>>> PHASE 4: Restoring standard application HNSW index...")
            self.restore_default_index()

        summary_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": sys_meta,
            "benchmark_parameters": {
                "top_k": top_k,
                "iterations_per_query": iterations_per_query,
                "num_queries": len(test_queries),
                "hnsw_m": hnsw_m,
                "hnsw_ef_construction": hnsw_ef_construction,
                "hnsw_ef_search_grid": ef_search_grid,
                "ivfflat_lists": ivfflat_lists,
                "ivfflat_probes_grid": probes_grid
            },
            "configurations_evaluated": len(experiment_results),
            "results": experiment_results
        }

        return summary_report
