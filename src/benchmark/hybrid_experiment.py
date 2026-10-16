"""
Controlled Hybrid Relational–Vector Selectivity Experiment Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Investigates empirical interactions between relational WHERE filtering and vector retrieval strategies:
  1. Exact Filtered Flat Search (Ground-truth sequential scan with relational filters)
  2. Filtered HNSW Graph Search (evaluating iterative_scan 'off' vs 'relaxed_order' & ef_search)
  3. Filtered IVFFlat Centroid Search (evaluating iterative_scan 'off' vs 'relaxed_order' & probes)

Evaluates 4 Selectivity Tiers:
  - Low Selectivity (Broad: ~60-80% matching rows, e.g. adult patients across major departments)
  - Moderate Selectivity (~15-25% matching rows, e.g. department = 'Cardiology')
  - High Selectivity (~1-3% matching rows, e.g. department = 'Pulmonology' AND doc_type = 'radiology_report')
  - Zero-Match / Ultra-Restrictive (0% matching rows, e.g. impossible age min_age = 150)

Measures:
  - Query Latency Distribution (Mean, Min, Max, P50, P95, P99 ms)
  - Throughput (QPS)
  - Filtered Ground Truth Recall@K (Distance-based 1-Recall and Strict ID-Recall)
  - Qualifying Result Count Deficit (instances where approximate index returns fewer results than exact)
  - Physical Execution Plan Inspection via EXPLAIN (ANALYZE, BUFFERS)
"""

import time
import math
import statistics
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder
from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.benchmark.index_experiment import VectorIndexExperimentRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark.hybrid_selectivity")

# Controlled selectivity tier configurations
SELECTIVITY_TIERS = {
    "tier1_low_selectivity": {
        "title": "Low Selectivity (Broad Filter ~70%)",
        "description": "Adult patients (age >= 18) in major departments",
        "filters": {
            "min_age": 18,
            "department": ["Cardiology", "Internal Medicine", "Pulmonology", "Emergency", "Neurology"]
        }
    },
    "tier2_moderate_selectivity": {
        "title": "Moderate Selectivity (~20%)",
        "description": "Cardiology department encounters",
        "filters": {
            "department": "Cardiology"
        }
    },
    "tier3_high_selectivity": {
        "title": "High Selectivity / Restrictive (~1.5%)",
        "description": "Pulmonology radiology reports",
        "filters": {
            "department": "Pulmonology",
            "document_type": "radiology_report"
        }
    },
    "tier4_zero_match": {
        "title": "Zero-Match / Ultra-Restrictive (0%)",
        "description": "Impossible age range (age >= 150)",
        "filters": {
            "min_age": 150
        }
    }
}

DEFAULT_HYBRID_QUERIES = [
    "acute myocardial infarction and coronary artery stenosis",
    "congestive heart failure with pulmonary edema and bilateral pleural effusion",
    "chronic obstructive pulmonary disease with acute respiratory distress",
    "type 2 diabetes mellitus with diabetic peripheral neuropathy",
    "ischemic stroke with focal neurological deficit and left hemiparesis"
]

class HybridSelectivityExperimentRunner:
    """
    Executes controlled empirical evaluations of relational filter interaction with vector indexes.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        expected_dimension: int = 384,
        embedder: Optional[TextEmbedder] = None
    ):
        self.model_id = model_id
        self.expected_dimension = expected_dimension
        self.embedder = embedder or TextEmbedder(model_name=model_id, dimension=expected_dimension)
        self.search_engine = PostgresHybridSearchEngine(
            model_id=model_id,
            expected_dimension=expected_dimension,
            embedder=self.embedder
        )
        self.index_runner = VectorIndexExperimentRunner(
            model_id=model_id,
            expected_dimension=expected_dimension,
            embedder=self.embedder
        )

    def measure_filter_corpus_selectivity(self, filters: Dict[str, Any]) -> Tuple[int, int, float]:
        """
        Determines the exact number of matching chunk vectors for given filters vs total corpus.
        Returns (matching_chunks_count, total_corpus_chunks, selectivity_percentage).
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = %s;",
                    (self.model_id,)
                )
                total_chunks = cur.fetchone()[0]

                # Run count query with same where clauses
                where_clauses = ["ce.model_id = %s"]
                params: List[Any] = [self.model_id]

                dept = filters.get("department")
                if dept:
                    if isinstance(dept, (list, tuple, set)):
                        where_clauses.append("e.department = ANY(%s)")
                        params.append(list(dept))
                    else:
                        where_clauses.append("e.department = %s")
                        params.append(dept)

                doc_type = filters.get("document_type")
                if doc_type:
                    if isinstance(doc_type, (list, tuple, set)):
                        where_clauses.append("cd.document_type = ANY(%s)")
                        params.append(list(doc_type))
                    else:
                        where_clauses.append("cd.document_type = %s")
                        params.append(doc_type)

                min_age = filters.get("min_age")
                if min_age is not None:
                    where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) >= %s")
                    params.append(int(min_age))

                max_age = filters.get("max_age")
                if max_age is not None:
                    where_clauses.append("EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) <= %s")
                    params.append(int(max_age))

                count_sql = f"""
                    SELECT COUNT(DISTINCT ce.chunk_id)
                    FROM chunk_embeddings ce
                    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    JOIN clinical_documents cd ON dc.document_id = cd.document_id
                    JOIN encounters e ON cd.encounter_id = e.encounter_id
                    JOIN patients p ON cd.patient_id = p.patient_id
                    WHERE {" AND ".join(where_clauses)};
                """
                cur.execute(count_sql, tuple(params))
                matching_count = cur.fetchone()[0]

        selectivity_pct = (matching_count / total_chunks * 100.0) if total_chunks > 0 else 0.0
        return matching_count, total_chunks, round(selectivity_pct, 2)

    @staticmethod
    def calculate_recalls(
        retrieved_items: List[Dict[str, Any]],
        ground_truth_items: List[Dict[str, Any]],
        k: int,
        epsilon: float = 1e-5
    ) -> Tuple[float, float]:
        """
        Calculates Distance 1-Recall@K and Strict ID Recall@K against filtered exact ground truth.
        """
        if not ground_truth_items:
            # If ground truth is empty (zero matching rows), returning empty is 100% accurate
            return (1.0, 1.0) if not retrieved_items else (0.0, 0.0)

        actual_k = min(k, len(ground_truth_items))
        if actual_k == 0:
            return 1.0, 1.0

        retrieved_ids = set(r["chunk_id"] for r in retrieved_items[:k])
        gt_ids = set(r["chunk_id"] for r in ground_truth_items[:k])
        strict_id_recall = len(retrieved_ids.intersection(gt_ids)) / float(actual_k)

        # Distance threshold D_K from ground truth
        dk_threshold = ground_truth_items[actual_k - 1]["cosine_distance"]
        dist_hits = sum(1 for r in retrieved_items[:k] if r["cosine_distance"] <= dk_threshold + epsilon)
        distance_recall = dist_hits / float(actual_k)

        return distance_recall, strict_id_recall

    def run_tier_evaluation(
        self,
        tier_key: str,
        queries: List[str],
        top_k: int = 10,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates Exact Flat, HNSW, and IVFFlat under a specific selectivity tier.
        """
        tier_cfg = SELECTIVITY_TIERS[tier_key]
        filters = tier_cfg["filters"]
        match_count, total_count, sel_pct = self.measure_filter_corpus_selectivity(filters)

        logger.info("-" * 78)
        logger.info(f"Evaluating {tier_cfg['title']}: {match_count}/{total_count} chunks ({sel_pct}%)")
        logger.info("-" * 78)

        # ---------------------------------------------------------------------
        # 1. Generate Filtered Ground Truth (Exact Flat Sequential Scan)
        # ---------------------------------------------------------------------
        ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        for q in queries:
            gt_res = self.search_engine.search(
                query=q,
                top_k=top_k,
                search_mode="exact_flat",
                explain=False,
                **filters
            )
            ground_truth[q] = gt_res["results"]

        # Define configurations to test under this selectivity tier
        eval_configs = [
            {
                "name": "Filtered Exact Flat Search",
                "strategy": "exact_flat",
                "index_type": "Flat",
                "session_params": None,
                "needs_index": None
            },
            {
                "name": "HNSW (ef_s=40, iterative=off)",
                "strategy": "hnsw",
                "index_type": "HNSW",
                "session_params": {"hnsw.ef_search": 40, "hnsw.iterative_scan": "off"},
                "needs_index": "hnsw"
            },
            {
                "name": "HNSW (ef_s=40, iterative=relaxed_order)",
                "strategy": "hnsw",
                "index_type": "HNSW",
                "session_params": {"hnsw.ef_search": 40, "hnsw.iterative_scan": "relaxed_order"},
                "needs_index": "hnsw"
            },
            {
                "name": "IVFFlat (prb=5, iterative=off)",
                "strategy": "ivfflat",
                "index_type": "IVFFlat",
                "session_params": {"ivfflat.probes": 5, "ivfflat.iterative_scan": "off"},
                "needs_index": "ivfflat"
            },
            {
                "name": "IVFFlat (prb=5, iterative=relaxed_order)",
                "strategy": "ivfflat",
                "index_type": "IVFFlat",
                "session_params": {"ivfflat.probes": 5, "ivfflat.iterative_scan": "relaxed_order"},
                "needs_index": "ivfflat"
            },
            {
                "name": "Planner Free Choice (Joint Query)",
                "strategy": "planner_choice",
                "index_type": "Joint Cost Optimization",
                "session_params": None,
                "needs_index": "hnsw"
            }
        ]

        tier_results = []

        current_index_state = None

        for cfg in eval_configs:
            req_idx = cfg["needs_index"]

            # Manage index state
            if req_idx == "ivfflat" and current_index_state != "ivfflat":
                self.index_runner.create_ivfflat_index(lists=25)
                current_index_state = "ivfflat"
            elif (req_idx == "hnsw" or req_idx is None) and current_index_state != "hnsw":
                self.index_runner.create_hnsw_index(m=16, ef_construction=64)
                current_index_state = "hnsw"

            latencies_ms: List[float] = []
            dist_recalls: List[float] = []
            id_recalls: List[float] = []
            returned_counts: List[int] = []
            plan_lines: List[str] = []
            plan_summary: Dict[str, Any] = {}

            # Execute timed iterations across query set
            for q_idx, q in enumerate(queries):
                # First run with explain to capture execution plan
                first_run = self.search_engine.search(
                    query=q,
                    top_k=top_k,
                    search_mode=cfg["strategy"],
                    session_params=cfg["session_params"],
                    explain=(q_idx == 0),
                    **filters
                )

                if q_idx == 0:
                    plan_lines = first_run["query_plan"]
                    plan_summary = first_run["plan_summary"]

                latencies_ms.append(first_run["database_time_ms"])
                returned_counts.append(first_run["returned_count"])

                gt = ground_truth.get(q, [])
                d_rec, id_rec = self.calculate_recalls(first_run["results"], gt, top_k)
                dist_recalls.append(d_rec)
                id_recalls.append(id_rec)

                # Subsequent timing iterations
                for _ in range(iterations - 1):
                    sub_run = self.search_engine.search(
                        query=q,
                        top_k=top_k,
                        search_mode=cfg["strategy"],
                        session_params=cfg["session_params"],
                        explain=False,
                        **filters
                    )
                    latencies_ms.append(sub_run["database_time_ms"])

            # Compute stats
            latencies_ms.sort()
            mean_lat = statistics.mean(latencies_ms)
            p50 = latencies_ms[int(math.ceil(0.50 * len(latencies_ms))) - 1]
            p95 = latencies_ms[int(math.ceil(0.95 * len(latencies_ms))) - 1]
            p99 = latencies_ms[int(math.ceil(0.99 * len(latencies_ms))) - 1]
            mean_dist_recall = statistics.mean(dist_recalls) if dist_recalls else 0.0
            mean_id_recall = statistics.mean(id_recalls) if id_recalls else 0.0
            avg_returned = statistics.mean(returned_counts) if returned_counts else 0.0

            # Calculate Result Count Deficit (if ANN returned fewer than exact search)
            gt_counts = [len(ground_truth[q]) for q in queries]
            avg_gt_count = statistics.mean(gt_counts) if gt_counts else 0.0
            deficit_count = max(0, avg_gt_count - avg_returned)

            tier_results.append({
                "config_name": cfg["name"],
                "index_type": cfg["index_type"],
                "strategy": cfg["strategy"],
                "planner_strategy": plan_summary.get("strategy", "unknown"),
                "scanned_indexes": plan_summary.get("indexes_scanned", []),
                "scanned_bitmap_indexes": plan_summary.get("bitmap_indexes_scanned", []),
                "latency_ms": {
                    "mean": round(mean_lat, 3),
                    "p50": round(p50, 3),
                    "p95": round(p95, 3),
                    "p99": round(p99, 3)
                },
                "qps": round(len(latencies_ms) / (sum(latencies_ms) / 1000.0), 2) if sum(latencies_ms) > 0 else 0.0,
                "recall_at_k": round(mean_dist_recall, 4),
                "strict_id_recall_at_k": round(mean_id_recall, 4),
                "avg_returned_count": round(avg_returned, 1),
                "expected_gt_count": round(avg_gt_count, 1),
                "result_deficit": round(deficit_count, 1),
                "sample_plan": plan_lines[:12]
            })

        return {
            "tier_key": tier_key,
            "tier_title": tier_cfg["title"],
            "tier_description": tier_cfg["description"],
            "filters": filters,
            "matching_chunks": match_count,
            "total_chunks": total_count,
            "selectivity_percentage": sel_pct,
            "configurations": tier_results
        }

    def run_full_hybrid_experiment(
        self,
        queries: Optional[List[str]] = None,
        top_k: int = 10,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Executes multi-selectivity controlled hybrid experiment across all 4 tiers.
        """
        test_queries = queries or DEFAULT_HYBRID_QUERIES
        logger.info("=" * 78)
        logger.info("  STARTING HYBRID RELATIONAL–VECTOR SELECTIVITY BENCHMARK")
        logger.info("=" * 78)

        sys_meta = self.index_runner.get_system_metadata()
        all_tier_results = []

        try:
            for tier_key in SELECTIVITY_TIERS:
                res = self.run_tier_evaluation(
                    tier_key=tier_key,
                    queries=test_queries,
                    top_k=top_k,
                    iterations=iterations
                )
                all_tier_results.append(res)
        finally:
            self.index_runner.restore_default_index()

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": sys_meta,
            "benchmark_parameters": {
                "top_k": top_k,
                "iterations_per_query": iterations,
                "num_queries": len(test_queries)
            },
            "tiers_evaluated": all_tier_results
        }
