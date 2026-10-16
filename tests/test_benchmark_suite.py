"""
Automated Test Suite for Database Benchmarking Engine & Telemetry
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Validates:
  - System hardware and PostgreSQL/pgvector telemetry extraction
  - Statistical latency distributions (mean, stddev, P50, P90, P95, P99, CV)
  - Noise and variance detection (STABLE, HIGH_VARIANCE, SMALL_SAMPLE_SIZE)
  - Recall@K computation (distance-based and ID-based)
  - End-to-end unified runner execution in smoke-test mode
  - Vector SVG plot generator output validity
"""

import math
import pytest
from pathlib import Path
from typing import Dict, Any, List

from src.benchmark.system_telemetry import (
    get_full_benchmark_telemetry,
    get_hardware_info,
    get_database_info
)
from src.benchmark.metrics import (
    aggregate_latency_distribution,
    calculate_recall_at_k
)
from src.benchmark.unified_runner import UnifiedBenchmarkRunner
from scripts.plot_benchmarks import render_svg_barchart

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_hardware_telemetry():
    """Verifies non-fabricated hardware telemetry extraction."""
    hw = get_hardware_info()
    assert isinstance(hw["processor"], str)
    assert hw["cpu_logical_cores"] > 0
    assert hw["total_ram_gb"] is not None and hw["total_ram_gb"] > 0
    assert hw["os_system"] == "Windows"

def test_database_pgvector_telemetry():
    """Verifies live PostgreSQL and pgvector version extraction."""
    db = get_database_info()
    assert "PostgreSQL" in db["postgres_version"]
    assert isinstance(db["pgvector_version"], str)
    assert len(db["pgvector_version"]) > 0
    assert db["embedding_model"]["dimension"] == 384
    assert db["embedding_model"]["distance_metric"] == "cosine"
    assert db["dataset_statistics"]["total_chunk_embeddings"] > 0

def test_full_benchmark_telemetry():
    """Verifies consolidated system telemetry payload."""
    telemetry = get_full_benchmark_telemetry(model_id="sentence-transformers/all-MiniLM-L6-v2")
    assert "hardware" in telemetry
    assert "database" in telemetry
    assert telemetry["database"]["embedding_model"]["dimension"] == 384
    assert telemetry["database"]["embedding_model"]["pgvector_operator"] == "<=>"

def test_metrics_percentile_calculation():
    """Verifies accurate calculation of percentiles, mean, stddev, and min/max."""
    samples = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    stats = aggregate_latency_distribution(samples)
    
    assert stats["sample_count"] == 10
    assert stats["min"] == 10.0
    assert stats["max"] == 28.0
    assert stats["mean"] == 19.0
    assert stats["p50"] == 18.0 or stats["p50"] == 19.0 or stats["p50"] == 20.0
    assert stats["p90"] >= 26.0
    assert stats["p95"] >= 26.0
    assert stats["p99"] >= 26.0
    assert stats["noise_flag"] == "STABLE"

def test_metrics_noise_detection():
    """Verifies noisy/high-variance flags and small sample size warnings."""
    # 1. High variance (CV > 0.50)
    noisy_samples = [1.0, 100.0, 2.0, 95.0, 1.5, 98.0]
    stats_noisy = aggregate_latency_distribution(noisy_samples)
    assert "HIGH_VARIANCE" in stats_noisy["noise_flag"]

    # 2. Small sample size (< 5)
    small_samples = [12.0, 12.5, 12.1]
    stats_small = aggregate_latency_distribution(small_samples)
    assert "SMALL_SAMPLE_SIZE" in stats_small["noise_flag"]

    # 3. Empty sample list
    stats_empty = aggregate_latency_distribution([])
    assert stats_empty["sample_count"] == 0
    assert stats_empty["mean"] == 0.0
    assert stats_empty["noise_flag"] == "NO_DATA"

def test_recall_at_k_calculation():
    """Verifies exact distance and ID recall calculations against ground truth."""
    gt = [
        {"chunk_id": "c1", "cosine_distance": 0.1000},
        {"chunk_id": "c2", "cosine_distance": 0.2000},
        {"chunk_id": "c3", "cosine_distance": 0.3000},
    ]
    # Exact match
    cand_exact = [
        {"chunk_id": "c1", "cosine_distance": 0.1000},
        {"chunk_id": "c2", "cosine_distance": 0.2000},
        {"chunk_id": "c3", "cosine_distance": 0.3000},
    ]
    d_rec, id_rec = calculate_recall_at_k(cand_exact, gt, k=3)
    assert d_rec == 1.0
    assert id_rec == 1.0

    # Partial ID match with distance tolerance
    cand_partial = [
        {"chunk_id": "c1", "cosine_distance": 0.100005},
        {"chunk_id": "c99", "cosine_distance": 0.4500},
        {"chunk_id": "c3", "cosine_distance": 0.29999},
    ]
    d_rec2, id_rec2 = calculate_recall_at_k(cand_partial, gt, k=3)
    assert id_rec2 == pytest.approx(2.0 / 3.0, 0.01)
    assert d_rec2 == pytest.approx(2.0 / 3.0, 0.01)

    # Empty ground truth
    d_empty, id_empty = calculate_recall_at_k([], [], k=3)
    assert d_empty == 1.0
    assert id_empty == 1.0

def test_svg_plot_generator():
    """Verifies pure-Python SVG vector rendering."""
    categories = ["Exact Flat", "HNSW (ef=40)", "IVFFlat (probes=5)"]
    latencies = [12.4, 3.2, 5.1]
    recalls = [1.0, 0.98, 0.92]

    svg_str = render_svg_barchart(
        title="Test Latency vs Recall",
        categories=categories,
        series_a=latencies,
        series_a_label="Latency (ms)",
        series_b=recalls,
        series_b_label="Recall@10",
        y_max_b=1.05
    )

    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert "Test Latency vs Recall" in svg_str
    assert "Exact Flat" in svg_str
    assert "<rect" in svg_str
    assert "<text" in svg_str

def test_unified_runner_smoke():
    """Verifies unified benchmark runner execution in fast smoke mode."""
    runner = UnifiedBenchmarkRunner()
    report = runner.run_full_suite(
        queries=["acute myocardial infarction"],
        top_k=5,
        warmup_runs=0,
        iterations=1,
        measure_cold=False,
        include_hybrid=False,
        hnsw_ef_search_values=[10],
        ivfflat_probes_values=[1]
    )

    assert "telemetry" in report
    assert "summary" in report
    assert "raw_samples" in report
    assert len(report["summary"]) >= 1

    flat_summary = next((s for s in report["summary"] if s["index_type"] == "Flat"), None)
    assert flat_summary is not None
    assert flat_summary["distance_recall"] == 1.0
    assert flat_summary["id_recall"] == 1.0
    assert flat_summary["warm_latency"]["p50"] > 0.0
