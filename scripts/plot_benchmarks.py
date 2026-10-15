"""
Benchmark Visualization Script
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Renders academic comparison charts from empirical benchmark results into results/benchmark_charts.png.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results(results_dir: Path):
    json_path = results_dir / "benchmark_metrics.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx_data = data["indexing_benchmarks"]
    hyb_data = data["hybrid_query_benchmarks"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.3)

    # Chart 1: Indexing Strategies - Latency vs Recall
    strategies = [m["strategy"].split("(")[0].strip() for m in idx_data]
    latencies = [m["mean_latency_ms"] for m in idx_data]
    recalls = [m["recall_at_10"] for m in idx_data]

    x = np.arange(len(strategies))
    width = 0.35

    color_lat = "#1f77b4"
    color_rec = "#2ca02c"

    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    b1 = ax1.bar(x - width/2, latencies, width, label="Mean Latency (ms)", color=color_lat)
    b2 = ax1_twin.bar(x + width/2, recalls, width, label="Recall@10", color=color_rec)

    ax1.set_ylabel("Latency (ms)", color=color_lat, fontsize=11, fontweight="bold")
    ax1_twin.set_ylabel("Recall@10", color=color_rec, fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=10)
    ax1.set_title("Indexing Strategy: Latency vs. Recall@10", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Chart 2: Hybrid Query Strategies - Latency Comparison
    hyb_strategies = [m["strategy"].replace(" (Attribute-First)", "").replace(" (Vector-First)", "") for m in hyb_data]
    hyb_latencies = [m["mean_latency_ms"] for m in hyb_data]
    colors = ["#7f7f7f", "#2ca02c", "#d62728"]

    ax2 = axes[1]
    bars = ax2.bar(hyb_strategies, hyb_latencies, color=colors, width=0.5)
    ax2.set_ylabel("Mean Query Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Hybrid Query Execution Latency (Cardiology Filter)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    out_file = results_dir / "benchmark_charts.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved benchmark chart visualization to {out_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    res_dir = base_dir / "results"
    plot_benchmark_results(res_dir)
