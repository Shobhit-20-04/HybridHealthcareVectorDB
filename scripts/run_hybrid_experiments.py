"""
Reproducible Hybrid Relational–Vector Selectivity Benchmark Harness
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Evaluates how relational WHERE filters interact with vector indexes (Flat, HNSW, IVFFlat)
across 4 selectivity tiers (Low, Moderate, High, Zero-Match). Measures latency distributions,
throughput (QPS), filtered ground-truth Recall@K, and detects result count deficits.
Exports structured benchmark results to results/hybrid_query_benchmark.json.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.hybrid_experiment import (
    HybridSelectivityExperimentRunner,
    DEFAULT_HYBRID_QUERIES
)

def format_tier_table(tier_data: dict) -> str:
    """
    Renders a Markdown-compatible table summarizing benchmark results for a single selectivity tier.
    """
    header = (
        f"### {tier_data['tier_title']} — Selectivity: {tier_data['selectivity_percentage']}% "
        f"({tier_data['matching_chunks']}/{tier_data['total_chunks']} chunks)\n"
        f"*Filter Predicates: {json.dumps(tier_data['filters'])}*\n\n"
        "| Configuration                             | Strategy          | Optimizer Strategy         | Mean (ms) | P50 (ms) | P95 (ms) | QPS    | 1-Recall | ID-Recall | Ret / Expected | Deficit |\n"
        "|:------------------------------------------|:------------------|:---------------------------|----------:|---------:|---------:|-------:|---------:|----------:|:--------------:|--------:|"
    )
    rows = []
    for cfg in tier_data["configurations"]:
        name = cfg["config_name"]
        strategy = cfg["strategy"]
        plan_strat = cfg["planner_strategy"]
        lat = cfg["latency_ms"]
        mean_lat = f"{lat['mean']:.3f}"
        p50 = f"{lat['p50']:.3f}"
        p95 = f"{lat['p95']:.3f}"
        qps = f"{cfg['qps']:.1f}"
        d_rec = f"{cfg['recall_at_k'] * 100.0:.2f}%"
        id_rec = f"{cfg['strict_id_recall_at_k'] * 100.0:.2f}%"
        counts = f"{cfg['avg_returned_count']:.0f} / {cfg['expected_gt_count']:.0f}"
        deficit = f"{cfg['result_deficit']:.0f}"

        rows.append(
            f"| {name:<41} | {strategy:<17} | {plan_strat:<26} | "
            f"{mean_lat:>9} | {p50:>8} | {p95:>8} | {qps:>6} | {d_rec:>8} | {id_rec:>9} | {counts:^14} | {deficit:>7} |"
        )

    return header + "\n" + "\n".join(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Controlled Hybrid Relational-Vector Selectivity Experiment Runner"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest documents to retrieve (default: 10)")
    parser.add_argument("--iterations", type=int, default=3, help="Timing iterations per query (default: 3)")
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/hybrid_query_benchmark.json", 
        help="Output JSON results path (default: results/hybrid_query_benchmark.json)"
    )
    parser.add_argument(
        "--limit-queries", 
        type=int, 
        default=None, 
        help="Optional limit on clinical queries for rapid smoke evaluation"
    )

    args = parser.parse_args()

    queries = DEFAULT_HYBRID_QUERIES
    if args.limit_queries is not None and args.limit_queries > 0:
        queries = queries[:args.limit_queries]

    runner = HybridSelectivityExperimentRunner()

    print("\n" + "=" * 120)
    print("  CONTROLLED HYBRID RELATIONAL–VECTOR SELECTIVITY BENCHMARK")
    print("=" * 120)
    sys_meta = runner.index_runner.get_system_metadata()
    print(f"[*] Engine / Database  : {sys_meta['postgres_version'].split(' on ')[0]}")
    print(f"[*] pgvector Version   : v{sys_meta['pgvector_version']}")
    print(f"[*] Model Registry      : {sys_meta['model_id']} ({sys_meta['dimension']}-dim)")
    print(f"[*] Total Chunk Corpus  : {sys_meta['model_vectors_count']} dense vectors ({sys_meta['table_disk_size_pretty']})")
    print(f"[*] Evaluation Depth    : {len(queries)} clinical queries | Top-K = {args.top_k} | Iterations = {args.iterations}")
    print("=" * 120)

    # Run experiments
    report = runner.run_full_hybrid_experiment(
        queries=queries,
        top_k=args.top_k,
        iterations=args.iterations
    )

    # Render summary tables
    print("\n" + "=" * 120)
    print("  EMPIRICAL HYBRID QUERY SELECTIVITY BENCHMARK RESULTS")
    print("=" * 120 + "\n")
    for tier in report["tiers_evaluated"]:
        table_str = format_tier_table(tier)
        print(table_str + "\n")
    print("=" * 120)

    # Export JSON
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[V] Hybrid selectivity benchmark completed successfully. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
