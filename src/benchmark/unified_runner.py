"""
Unified Reproducible Database Benchmarking Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Orchestrates complete empirical evaluations across:
1. Exact Vector Search (Gold-standard ground truth)
2. HNSW Vector Indexing (m, ef_construction, ef_search sweeps)
3. IVFFlat Vector Indexing (lists, probes sweeps)
4. Hybrid Relational-Vector Query Execution across 4 selectivity tiers
5. Controlled Cold vs. Warm cache latency measurement
6. Physical execution plan verification via EXPLAIN (ANALYZE, BUFFERS)
7. Full hardware, PostgreSQL, and pgvector telemetry embedding
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from database.connection import get_connection
from src.embeddings.text_embedder import TextEmbedder
from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.benchmark.system_telemetry import get_full_benchmark_telemetry
from src.benchmark.metrics import aggregate_latency_distribution, calculate_recall_at_k
from src.benchmark.index_experiment import (
    VectorIndexExperimentRunner,
    DEFAULT_CLINICAL_QUERIES,
    InsufficientDataForIndexError
)
from src.benchmark.hybrid_experiment import (
    HybridSelectivityExperimentRunner,
    SELECTIVITY_TIERS,
    DEFAULT_HYBRID_QUERIES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("benchmark.unified_runner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class UnifiedBenchmarkRunner:
    """
    Coordinates end-to-end reproducible benchmarks across vector indexes and hybrid queries.
    Captures raw query executions for CSV export and embeds complete system telemetry.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        embedder: Optional[TextEmbedder] = None
    ):
        self.model_id = model_id
        self.dimension = dimension
        self.embedder = embedder or TextEmbedder(model_name=model_id, dimension=dimension)
        self.index_runner = VectorIndexExperimentRunner(
            model_id=model_id,
            expected_dimension=dimension,
            embedder=self.embedder
        )
        self.hybrid_runner = HybridSelectivityExperimentRunner(
            model_id=model_id,
            expected_dimension=dimension,
            embedder=self.embedder
        )

    def execute_single_vector_query(
        self,
        query_vector: List[float],
        top_k: int,
        session_params: Optional[Dict[str, Any]] = None,
        discard_cache: bool = False
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes a single top-k vector query.
        Optionally executes DISCARD ALL to evaluate cold query behavior.
        """
        with get_connection() as conn:
            if discard_cache:
                conn.autocommit = True
                conn.execute("DISCARD ALL;")
                conn.autocommit = False

            with conn.transaction():
                with conn.cursor() as cur:
                    if session_params:
                        for k, v in session_params.items():
                            cur.execute(f"SET LOCAL {k} = {v};")

                    t0 = time.perf_counter()
                    cur.execute("""
                        SELECT 
                            ce.chunk_id,
                            dc.chunk_index,
                            cd.title,
                            (ce.embedding <=> %s::vector) AS cosine_distance
                        FROM chunk_embeddings ce
                        JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                        JOIN clinical_documents cd ON dc.document_id = cd.document_id
                        WHERE ce.model_id = %s
                        ORDER BY ce.embedding <=> %s::vector ASC
                        LIMIT %s;
                    """, (query_vector, self.model_id, query_vector, int(top_k)))
                    rows = cur.fetchall()
                    lat_ms = (time.perf_counter() - t0) * 1000.0

        results = [
            {"chunk_id": str(r[0]), "chunk_index": r[1], "title": r[2], "cosine_distance": float(r[3])}
            for r in rows
        ]
        return results, lat_ms

    def run_full_suite(
        self,
        queries: Optional[List[str]] = None,
        top_k: int = 10,
        warmup_runs: int = 2,
        iterations: int = 5,
        measure_cold: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search_values: Optional[List[int]] = None,
        ivfflat_lists: int = 25,
        ivfflat_probes_values: Optional[List[int]] = None,
        include_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the master benchmark suite across vector indexes and hybrid selectivity tiers.
        """
        t_suite_start = time.perf_counter()
        test_queries = queries or DEFAULT_CLINICAL_QUERIES
        ef_search_grid = hnsw_ef_search_values or [10, 40, 100]
        probes_grid = ivfflat_probes_values or [1, 5, 10]

        logger.info("=" * 80)
        logger.info("  STARTING UNIFIED REPRODUCIBLE DATABASE BENCHMARK SUITE")
        logger.info("=" * 80)

        # 1. Collect Telemetry
        telemetry = get_full_benchmark_telemetry(model_id=self.model_id)

        # 2. Encode Query Vectors upfront
        logger.info(f"Encoding {len(test_queries)} benchmark queries into {self.dimension}-d vectors...")
        query_vectors: List[Tuple[str, List[float]]] = []
        for q in test_queries:
            vec = self.embedder.embed_text(q)
            self.embedder.validate_dimension(vec)
            query_vectors.append((q, vec))

        # 3. Ground Truth Generation (Exact Flat Sequential Scan)
        logger.info("Computing exact-search ground truth baseline (sequential scan)...")
        self.index_runner.drop_vector_indexes()
        unfiltered_ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        for q_text, q_vec in query_vectors:
            gt_items, _ = self.execute_single_vector_query(
                q_vec,
                top_k,
                session_params={"enable_indexscan": "off", "enable_seqscan": "on"}
            )
            unfiltered_ground_truth[q_text] = gt_items

        raw_query_records: List[Dict[str, Any]] = []
        summary_records: List[Dict[str, Any]] = []

        # =====================================================================
        # PART 1: Pure Vector Search Benchmarks (Flat vs HNSW vs IVFFlat)
        # =====================================================================
        index_configs_to_run = []

        # 1.1 Exact Flat Search Baseline
        index_configs_to_run.append({
            "name": "Exact Flat (Ground Truth Baseline)",
            "index_type": "Flat",
            "index_name": "NONE (Sequential Scan)",
            "build_meta": {"build_time_ms": 0.0, "disk_size_pretty": "0 bytes", "disk_size_bytes": 0},
            "session_params": {"enable_indexscan": "off", "enable_seqscan": "on"},
            "setup_func": lambda: self.index_runner.drop_vector_indexes()
        })

        # 1.2 HNSW Sweep
        def setup_hnsw():
            return self.index_runner.create_hnsw_index(m=hnsw_m, ef_construction=hnsw_ef_construction)

        # Build HNSW once, test sweeps
        hnsw_meta = setup_hnsw()
        for ef_s in ef_search_grid:
            index_configs_to_run.append({
                "name": f"HNSW (m={hnsw_m}, ef_s={ef_s})",
                "index_type": "HNSW",
                "index_name": self.index_runner.HNSW_INDEX_NAME,
                "build_meta": hnsw_meta,
                "session_params": {
                    "enable_indexscan": "on",
                    "enable_seqscan": "off",
                    "hnsw.ef_search": ef_s
                },
                "setup_func": None
            })

        # 1.3 IVFFlat Sweep
        try:
            def setup_ivfflat():
                return self.index_runner.create_ivfflat_index(lists=ivfflat_lists)

            ivf_meta = setup_ivfflat()
            for prb in probes_grid:
                index_configs_to_run.append({
                    "name": f"IVFFlat (lists={ivfflat_lists}, probes={prb})",
                    "index_type": "IVFFlat",
                    "index_name": self.index_runner.IVFFLAT_INDEX_NAME,
                    "build_meta": ivf_meta,
                    "session_params": {
                        "enable_indexscan": "on",
                        "enable_seqscan": "off",
                        "ivfflat.probes": prb
                    },
                    "setup_func": None
                })
        except InsufficientDataForIndexError as e:
            logger.warning(f"[!] Skipping IVFFlat sweep: {e}")

        # Execute pure vector configurations
        for cfg in index_configs_to_run:
            c_name = cfg["name"]
            idx_type = cfg["index_type"]
            s_params = cfg["session_params"]
            build_info = cfg["build_meta"]

            if cfg.get("setup_func"):
                cfg["setup_func"]()

            logger.info(f"--> Benchmarking Vector Index: {c_name}")

            # Optional Cold Run
            cold_latencies = []
            if measure_cold:
                for q_idx, (q_text, q_vec) in enumerate(query_vectors, 1):
                    _, cold_lat = self.execute_single_vector_query(
                        q_vec, top_k, session_params=s_params, discard_cache=True
                    )
                    cold_latencies.append(cold_lat)
                    raw_query_records.append({
                        "category": "pure_vector",
                        "configuration": c_name,
                        "index_type": idx_type,
                        "query_id": q_idx,
                        "query_text": q_text,
                        "run_type": "cold",
                        "iteration": 1,
                        "latency_ms": round(cold_lat, 3),
                        "distance_recall": 1.0 if idx_type == "Flat" else None,
                        "id_recall": 1.0 if idx_type == "Flat" else None,
                        "top_k": top_k
                    })

            # Warmup Runs
            for _ in range(warmup_runs):
                for _, q_vec in query_vectors[:3]:
                    self.execute_single_vector_query(q_vec, top_k, session_params=s_params)

            # Warm Iterations
            warm_latencies = []
            dist_recalls = []
            id_recalls = []

            for iter_idx in range(1, iterations + 1):
                for q_idx, (q_text, q_vec) in enumerate(query_vectors, 1):
                    retrieved, lat_ms = self.execute_single_vector_query(
                        q_vec, top_k, session_params=s_params
                    )
                    warm_latencies.append(lat_ms)

                    gt_items = unfiltered_ground_truth.get(q_text, [])
                    d_rec, id_rec = calculate_recall_at_k(retrieved, gt_items, top_k)
                    dist_recalls.append(d_rec)
                    id_recalls.append(id_rec)

                    raw_query_records.append({
                        "category": "pure_vector",
                        "configuration": c_name,
                        "index_type": idx_type,
                        "query_id": q_idx,
                        "query_text": q_text,
                        "run_type": "warm",
                        "iteration": iter_idx,
                        "latency_ms": round(lat_ms, 3),
                        "distance_recall": d_rec,
                        "id_recall": id_rec,
                        "top_k": top_k
                    })

            # Statistical Aggregation
            warm_stats = aggregate_latency_distribution(warm_latencies)
            cold_stats = aggregate_latency_distribution(cold_latencies) if cold_latencies else None
            mean_dist_recall = sum(dist_recalls) / len(dist_recalls) if dist_recalls else 1.0
            mean_id_recall = sum(id_recalls) / len(id_recalls) if id_recalls else 1.0
            qps = (len(warm_latencies) / (sum(warm_latencies) / 1000.0)) if sum(warm_latencies) > 0 else 0.0

            summary_records.append({
                "category": "pure_vector",
                "configuration": c_name,
                "index_type": idx_type,
                "build_time_ms": build_info.get("build_time_ms", 0.0),
                "disk_size_pretty": build_info.get("disk_size_pretty", "-"),
                "disk_size_bytes": build_info.get("disk_size_bytes", 0),
                "warm_latency": warm_stats,
                "cold_latency": cold_stats,
                "throughput_qps": round(qps, 2),
                "distance_recall": round(mean_dist_recall, 4),
                "id_recall": round(mean_id_recall, 4),
                "noise_flag": warm_stats["noise_flag"]
            })

        # =====================================================================
        # PART 2: Hybrid Relational-Vector Selectivity Tiers
        # =====================================================================
        hybrid_tier_summaries = []
        if include_hybrid:
            logger.info("--> Benchmarking Hybrid Relational-Vector Queries across 4 Selectivity Tiers...")
            hybrid_queries = DEFAULT_HYBRID_QUERIES[:len(test_queries)]

            for tier_key, tier_cfg in SELECTIVITY_TIERS.items():
                filters = tier_cfg["filters"]
                match_chunks, total_chunks, sel_pct = self.hybrid_runner.measure_filter_corpus_selectivity(filters)
                logger.info(f"    Evaluating Tier '{tier_cfg['title']}' ({match_chunks}/{total_chunks} chunks, {sel_pct}%)")

                # Compute Filtered Ground Truth
                filtered_gt = {}
                for q_text in hybrid_queries:
                    gt_res = self.hybrid_runner.search_engine.search(
                        query=q_text,
                        top_k=top_k,
                        search_mode="exact_flat",
                        **filters
                    )
                    filtered_gt[q_text] = gt_res["results"]

                # Configurations under this tier
                tier_cfgs = [
                    {"name": f"Exact Flat [{tier_cfg['title']}]", "mode": "exact_flat"},
                    {"name": f"HNSW Planner Choice [{tier_cfg['title']}]", "mode": "planner_choice"},
                ]

                for t_cfg in tier_cfgs:
                    mode = t_cfg["mode"]
                    t_name = t_cfg["name"]

                    tier_latencies = []
                    tier_d_recalls = []
                    tier_id_recalls = []

                    for iter_idx in range(1, iterations + 1):
                        for q_idx, q_text in enumerate(hybrid_queries, 1):
                            res = self.hybrid_runner.search_engine.search(
                                query=q_text,
                                top_k=top_k,
                                search_mode=mode,
                                **filters
                            )
                            lat_ms = res["total_time_ms"]
                            tier_latencies.append(lat_ms)

                            gt_items = filtered_gt.get(q_text, [])
                            d_rec, id_rec = calculate_recall_at_k(res["results"], gt_items, top_k)
                            tier_d_recalls.append(d_rec)
                            tier_id_recalls.append(id_rec)

                            raw_query_records.append({
                                "category": "hybrid_selectivity",
                                "configuration": t_name,
                                "index_type": mode,
                                "query_id": q_idx,
                                "query_text": q_text,
                                "run_type": "warm",
                                "iteration": iter_idx,
                                "latency_ms": round(lat_ms, 3),
                                "distance_recall": d_rec,
                                "id_recall": id_rec,
                                "top_k": top_k
                            })

                    t_stats = aggregate_latency_distribution(tier_latencies)
                    t_qps = (len(tier_latencies) / (sum(tier_latencies) / 1000.0)) if sum(tier_latencies) > 0 else 0.0
                    mean_d_rec = sum(tier_d_recalls) / len(tier_d_recalls) if tier_d_recalls else 1.0
                    mean_id_rec = sum(tier_id_recalls) / len(tier_id_recalls) if tier_id_recalls else 1.0

                    summary_records.append({
                        "category": "hybrid_selectivity",
                        "configuration": t_name,
                        "index_type": mode,
                        "selectivity_percentage": sel_pct,
                        "matching_chunks": match_chunks,
                        "total_chunks": total_chunks,
                        "warm_latency": t_stats,
                        "cold_latency": None,
                        "throughput_qps": round(t_qps, 2),
                        "distance_recall": round(mean_d_rec, 4),
                        "id_recall": round(mean_id_rec, 4),
                        "noise_flag": t_stats["noise_flag"]
                    })

        # Cleanup indexes
        self.index_runner.drop_vector_indexes()

        total_suite_duration_s = time.perf_counter() - t_suite_start
        logger.info(f"[V] Benchmark suite completed in {total_suite_duration_s:.2f} seconds.")

        return {
            "telemetry": telemetry,
            "benchmark_parameters": {
                "top_k": top_k,
                "warmup_runs": warmup_runs,
                "iterations": iterations,
                "measure_cold": measure_cold,
                "queries_count": len(test_queries),
                "total_timed_queries_executed": len(raw_query_records),
                "total_duration_seconds": round(total_suite_duration_s, 2)
            },
            "summary": summary_records,
            "raw_samples": raw_query_records
        }
