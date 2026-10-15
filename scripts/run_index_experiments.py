"""
Reproducible Vector Index Benchmarking CLI Harness
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes controlled empirical evaluations across Flat, HNSW, and IVFFlat vector indexes
in PostgreSQL with pgvector. Measures build time, memory footprint, latency distributions,
throughput (QPS), and empirical Recall@K against exact sequential scan ground truth.
Outputs markdown-compatible comparison tables and exports structured JSON benchmark reports.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.index_experiment import (
    VectorIndexExperimentRunner,
    DEFAULT_CLINICAL_QUERIES
)

def format_benchmark_table(report: dict) -> str:
    """
    Renders a Markdown-compatible comparison table summarizing all index configurations.
    """
    header = (
        "| Configuration                     | Index Type        | Build (ms) | Size    | Plan OK | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | QPS     | 1-Recall | ID-Recall |\n"
        "|:----------------------------------|:------------------|-----------:|--------:|:-------:|----------:|---------:|---------:|---------:|--------:|---------:|----------:|"
    )
    rows = []
    top_k = report["benchmark_parameters"]["top_k"]

    for res in report["results"]:
        name = res["config_name"]
        idx_type = res["index_type"]
        build_ms = f"{res['build_time_ms']:.1f}" if res["build_time_ms"] > 0 else "-"
        size_str = res["disk_size_pretty"]
        plan_ok = "[V] YES" if res["plan_verified"] else "[!] NO"
        lat = res["latency_ms"]
        mean_lat = f"{lat['mean']:.3f}"
        p50_lat = f"{lat['p50']:.3f}"
        p95_lat = f"{lat['p95']:.3f}"
        p99_lat = f"{lat['p99']:.3f}"
        qps = f"{res['throughput_qps']:.1f}"
        dist_recall = f"{res['recall_at_k'] * 100.0:.2f}%"
        id_recall = f"{res.get('strict_id_recall_at_k', 0.0) * 100.0:.2f}%"

        rows.append(
            f"| {name:<33} | {idx_type:<17} | {build_ms:>10} | {size_str:>7} | {plan_ok:^7} | "
            f"{mean_lat:>9} | {p50_lat:>8} | {p95_lat:>8} | {p99_lat:>8} | {qps:>7} | {dist_recall:>8} | {id_recall:>9} |"
        )

    return header + "\n" + "\n".join(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Controlled Vector Index Benchmark Harness for PostgreSQL / pgvector"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest neighbors to retrieve (default: 10)")
    parser.add_argument("--iterations", type=int, default=3, help="Timing iterations per query (default: 3)")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW m connections per node (default: 16)")
    parser.add_argument("--hnsw-ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64)")
    parser.add_argument(
        "--hnsw-ef-search", 
        type=str, 
        default="10,40,100", 
        help="Comma-separated HNSW ef_search sweep values (default: '10,40,100')"
    )
    parser.add_argument("--ivfflat-lists", type=int, default=25, help="IVFFlat Voronoi centroid count (default: 25)")
    parser.add_argument(
        "--ivfflat-probes", 
        type=str, 
        default="1,5,10", 
        help="Comma-separated IVFFlat probes sweep values (default: '1,5,10')"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/vector_index_benchmark.json", 
        help="Output JSON results path (default: results/vector_index_benchmark.json)"
    )
    parser.add_argument(
        "--limit-queries", 
        type=int, 
        default=None, 
        help="Optional limit on clinical queries for rapid smoke evaluation"
    )
    args = parser.parse_args()

    # Parse sweep parameters
    hnsw_ef_search_vals = [int(x.strip()) for x in args.hnsw_ef_search.split(",") if x.strip()]
    ivfflat_probes_vals = [int(x.strip()) for x in args.ivfflat_probes.split(",") if x.strip()]

    # Query subset
    queries = DEFAULT_CLINICAL_QUERIES
    if args.limit_queries is not None and args.limit_queries > 0:
        queries = queries[:args.limit_queries]

    runner = VectorIndexExperimentRunner()

    print("\n" + "=" * 115)
    print("  CONTROLLED VECTOR INDEX EMPIRICAL BENCHMARK: POSTGRESQL & PGVECTOR")
    print("=" * 115)
    sys_meta = runner.get_system_metadata()
    print(f"[*] Engine / Database : {sys_meta['postgres_version'].split(' on ')[0]}")
    print(f"[*] pgvector Version  : v{sys_meta['pgvector_version']}")
    print(f"[*] Embedding Model   : {sys_meta['model_id']} ({sys_meta['dimension']}-dim, Cosine Distance <=>)")
    print(f"[*] Corpus Size       : {sys_meta['model_vectors_count']} dense chunk vectors ({sys_meta['table_disk_size_pretty']})")
    print(f"[*] Evaluation Suite  : {len(queries)} clinical queries | Top-K = {args.top_k} | Iterations = {args.iterations}")
    print(f"[*] HNSW Sweep Grid   : m={args.hnsw_m}, ef_construction={args.hnsw_ef_construction}, ef_search={hnsw_ef_search_vals}")
    print(f"[*] IVFFlat Sweep Grid: lists={args.ivfflat_lists}, probes={ivfflat_probes_vals}")
    print("=" * 115)

    # Run experiment
    report = runner.run_controlled_experiment(
        queries=queries,
        top_k=args.top_k,
        iterations_per_query=args.iterations,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search_values=hnsw_ef_search_vals,
        ivfflat_lists=args.ivfflat_lists,
        ivfflat_probes_values=ivfflat_probes_vals
    )

    # Render results table
    print("\n" + "=" * 115)
    print("  EMPIRICAL BENCHMARK RESULTS TABLE")
    print("=" * 115)
    table_str = format_benchmark_table(report)
    print(table_str)
    print("=" * 115)

    # Save output JSON
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[V] Benchmark completed successfully. Results exported to: {output_path}")

if __name__ == "__main__":
    main()
