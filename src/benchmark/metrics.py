"""
Benchmark Metrics Calculation & Statistical Aggregation
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. Exact and approximate latency percentiles (P50, P90, P95, P99), mean, stddev, min, max
2. Recall@K calculation (Distance-based 1-Recall@K and Strict ID-based Recall@K)
3. Throughput calculation (Queries Per Second / QPS)
4. Variance and Noise Detection (flags high-variance or inconclusive runs)
"""

import math
import statistics
from typing import List, Dict, Any, Tuple, Optional

def calculate_percentile(sorted_data: List[float], p: float) -> float:
    """Calculates empirical percentile on pre-sorted numeric list."""
    if not sorted_data:
        return 0.0
    idx = int(math.ceil(p * len(sorted_data))) - 1
    return sorted_data[max(0, min(idx, len(sorted_data) - 1))]

def aggregate_latency_distribution(latencies_ms: List[float]) -> Dict[str, Any]:
    """
    Computes statistical distribution over latency measurements.
    Returns mean, stddev, median (P50), P90, P95, P99, min, and max.
    """
    if not latencies_ms:
        return {
            "sample_count": 0,
            "mean": 0.0,
            "stddev": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "cv": 0.0,
            "noise_flag": "NO_DATA"
        }

    sorted_vals = sorted(latencies_ms)
    count = len(sorted_vals)
    mean_val = statistics.mean(sorted_vals)
    stddev_val = statistics.stdev(sorted_vals) if count > 1 else 0.0
    cv = (stddev_val / mean_val) if mean_val > 0 else 0.0

    p50 = calculate_percentile(sorted_vals, 0.50)
    p90 = calculate_percentile(sorted_vals, 0.90)
    p95 = calculate_percentile(sorted_vals, 0.95)
    p99 = calculate_percentile(sorted_vals, 0.99)
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Noise analysis
    flags = []
    if count < 5:
        flags.append("SMALL_SAMPLE_SIZE")
    if cv > 0.50:
        flags.append("HIGH_VARIANCE")
    if cv > 1.0:
        flags.append("NOISY_MEASUREMENT")

    noise_flag = "; ".join(flags) if flags else "STABLE"

    return {
        "sample_count": count,
        "mean": round(mean_val, 3),
        "stddev": round(stddev_val, 3),
        "cv": round(cv, 3),
        "p50": round(p50, 3),
        "p90": round(p90, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "min": round(min_val, 3),
        "max": round(max_val, 3),
        "noise_flag": noise_flag
    }

def calculate_recall_at_k(
    retrieved_items: List[Dict[str, Any]],
    ground_truth_items: List[Dict[str, Any]],
    k: int,
    epsilon: float = 1e-5
) -> Tuple[float, float]:
    """
    Computes empirical accuracy relative to exact-search ground truth:
    1. Distance 1-Recall@K: Fraction of items whose distance <= (D_k_gt + epsilon)
    2. Strict ID Recall@K: Fraction of matching chunk_id UUIDs in top-k
    """
    if not ground_truth_items:
        # If ground truth has 0 matching items (e.g. restrictive zero-match filter),
        # returning 0 items is 100% accurate.
        return (1.0, 1.0) if not retrieved_items else (0.0, 0.0)

    actual_k = min(k, len(ground_truth_items))
    if actual_k == 0:
        return 1.0, 1.0

    # 1. ID Overlap
    retrieved_ids = set(r.get("chunk_id") for r in retrieved_items[:k])
    gt_ids = set(r.get("chunk_id") for r in ground_truth_items[:k])
    id_recall = len(retrieved_ids.intersection(gt_ids)) / float(actual_k)

    # 2. Distance Equivalence Threshold
    gt_threshold = ground_truth_items[actual_k - 1].get("cosine_distance", 1.0)
    dist_hits = sum(
        1 for r in retrieved_items[:k]
        if r.get("cosine_distance", 2.0) <= (gt_threshold + epsilon)
    )
    dist_recall = dist_hits / float(actual_k)

    return round(dist_recall, 4), round(id_recall, 4)
