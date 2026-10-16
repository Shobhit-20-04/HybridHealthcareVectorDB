"""
Master Reproducible Database Benchmarking CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes full empirical benchmark suite across:
1. Exact Vector Search (Sequential Scan Ground Truth Baseline)
2. HNSW Vector Indexing (m, ef_construction, ef_search sweeps)
3. IVFFlat Vector Indexing (lists, probes sweeps)
4. Hybrid Relational–Vector Queries (across 4 Selectivity Tiers)
5. Cold vs. Warm cache timing analysis
6. Exports raw samples to CSV, summary table to CSV/MD, and telemetry to JSON.
"""

import sys
import csv
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.unified_runner import UnifiedBenchmarkRunner
from src.benchmark.index_experiment import DEFAULT_CLINICAL_QUERIES

def export_raw_queries_csv(raw_samples: list, output_file: Path, telemetry: dict) -> None:
    """Exports every individual query execution timing sample with telemetry headers."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hw = telemetry.get("hardware", {})
    db = telemetry.get("database", {})

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        # Prepend metadata comments
        f.write(f"# Benchmark Telemetry: {hw.get('processor')} | {hw.get('cpu_logical_cores')} cores | {hw.get('total_ram_gb')} GB RAM\n")
        f.write(f"# Database Engine: {db.get('postgres_version')}\n")
        f.write(f"# pgvector Version: v{db.get('pgvector_version')} | Metric: {db.get('embedding_model', {}).get('distance_metric')} ({db.get('embedding_model', {}).get('pgvector_operator')})\n")

        fieldnames = [
            "category", "configuration", "index_type", "query_id", "query_text",
            "run_type", "iteration", "latency_ms", "distance_recall", "id_recall", "top_k"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_samples:
            writer.writerow(row)

def export_summary_csv(summary_records: list, output_file: Path, telemetry: dict) -> None:
    """Exports aggregated configuration summary to CSV with full statistical distributions."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hw = telemetry.get("hardware", {})
    db = telemetry.get("database", {})

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Benchmark Summary: {hw.get('os_platform')} | PostgreSQL {db.get('postgres_version', '').split()[0]} | pgvector {db.get('pgvector_version')}\n")

        fieldnames = [
            "category", "configuration", "index_type", "build_time_ms", "disk_size_pretty",
            "mean_latency_ms", "stddev_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "min_latency_ms", "max_latency_ms", "cold_p50_ms", "throughput_qps",
            "distance_recall", "id_recall", "noise_flag"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in summary_records:
            w_lat = rec.get("warm_latency", {})
            c_lat = rec.get("cold_latency") or {}
            writer.writerow({
                "category": rec.get("category"),
                "configuration": rec.get("configuration"),
                "index_type": rec.get("index_type"),
                "build_time_ms": rec.get("build_time_ms", 0.0),
                "disk_size_pretty": rec.get("disk_size_pretty", "-"),
                "mean_latency_ms": w_lat.get("mean", 0.0),
                "stddev_ms": w_lat.get("stddev", 0.0),
                "p50_latency_ms": w_lat.get("p50", 0.0),
                "p90_latency_ms": w_lat.get("p90", 0.0),
                "p95_latency_ms": w_lat.get("p95", 0.0),
                "p99_latency_ms": w_lat.get("p99", 0.0),
                "min_latency_ms": w_lat.get("min", 0.0),
                "max_latency_ms": w_lat.get("max", 0.0),
                "cold_p50_ms": c_lat.get("p50", "-") if c_lat else "-",
                "throughput_qps": rec.get("throughput_qps", 0.0),
                "distance_recall": rec.get("distance_recall", 1.0),
                "id_recall": rec.get("id_recall", 1.0),
                "noise_flag": rec.get("noise_flag", "STABLE")
            })

def format_markdown_summary(report: dict) -> str:
    """Generates clean GitHub Flavored Markdown summary report."""
    telemetry = report["telemetry"]
    hw = telemetry.get("hardware", {})
    db = telemetry.get("database", {})
    params = report["benchmark_parameters"]

    md = []
    md.append("# Benchmark Evaluation Summary Report\n")
    md.append(f"**Hardware:** {hw.get('processor')} ({hw.get('cpu_logical_cores')} cores) | {hw.get('total_ram_gb')} GB RAM | {hw.get('os_platform')}  ")
    md.append(f"**Database:** {db.get('postgres_version', '').split(' on ')[0]} | `pgvector` v{db.get('pgvector_version')}  ")
    md.append(f"**Model & Space:** `{db.get('embedding_model', {}).get('model_id')}` ({db.get('embedding_model', {}).get('dimension')}-dim, Metric: `{db.get('embedding_model', {}).get('distance_metric')}` `{db.get('embedding_model', {}).get('pgvector_operator')}`)  ")
    md.append(f"**Parameters:** Top-K = {params['top_k']} | Warmup = {params['warmup_runs']} | Iterations = {params['iterations']} | Total Duration = {params['total_duration_seconds']}s\n")

    md.append("## Indexing Strategy & Query Performance Comparison\n")
    md.append("| Configuration | Index Type | Build Time | Size | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Cold P50 | QPS | 1-Recall | ID-Recall | Noise Status |")
    md.append("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")

    for rec in report["summary"]:
        w = rec.get("warm_latency", {})
        c = rec.get("cold_latency") or {}
        cold_str = f"{c.get('p50', '-')}" if c else "-"
        build_str = f"{rec.get('build_time_ms', 0.0):.1f} ms" if rec.get('build_time_ms', 0.0) > 0 else "-"

        md.append(
            f"| **{rec['configuration']}** | {rec['index_type']} | {build_str} | {rec.get('disk_size_pretty', '-')} | "
            f"{w.get('mean', 0.0):.3f} | {w.get('p50', 0.0):.3f} | {w.get('p90', 0.0):.3f} | {w.get('p95', 0.0):.3f} | {w.get('p99', 0.0):.3f} | "
            f"{cold_str} | {rec.get('throughput_qps', 0.0):.1f} | {rec.get('distance_recall', 1.0)*100.0:.2f}% | "
            f"{rec.get('id_recall', 1.0)*100.0:.2f}% | `{rec.get('noise_flag', 'STABLE')}` |"
        )

    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="Master Reproducible Database Benchmarking Harness")
    parser.add_argument("--top-k", type=int, default=10, help="Nearest neighbors to retrieve (default: 10)")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warm-up iterations (default: 2)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of timed measurement iterations (default: 5)")
    parser.add_argument("--limit-queries", type=int, default=None, help="Optional limit on clinical queries")
    parser.add_argument("--no-cold", action="store_true", help="Skip cold cache measurement")
    parser.add_argument("--no-hybrid", action="store_true", help="Skip hybrid selectivity benchmark")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW m graph connections (default: 16)")
    parser.add_argument("--hnsw-ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64)")
    parser.add_argument("--hnsw-ef-search", type=str, default="10,40,100", help="Comma-separated HNSW ef_search sweep")
    parser.add_argument("--ivfflat-lists", type=int, default=25, help="IVFFlat centroid clusters (default: 25)")
    parser.add_argument("--ivfflat-probes", type=str, default="1,5,10", help="Comma-separated IVFFlat probes sweep")
    parser.add_argument("--output-dir", type=str, default="results", help="Target output directory (default: results)")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test (limits to 2 queries, 1 iteration)")
    parser.add_argument("--plot", action="store_true", help="Render SVG and PNG comparison charts after benchmark")

    args = parser.parse_args()

    # Configure smoke test presets
    if args.smoke_test:
        print("[*] Smoke Test Mode Activated: Setting limit_queries=2, iterations=1, warmup=0, no_cold=True")
        limit_queries = 2
        iterations = 1
        warmup = 0
        measure_cold = False
        ef_search_vals = [40]
        probes_vals = [5]
    else:
        limit_queries = args.limit_queries
        iterations = args.iterations
        warmup = args.warmup
        measure_cold = not args.no_cold
        ef_search_vals = [int(x.strip()) for x in args.hnsw_ef_search.split(",") if x.strip()]
        probes_vals = [int(x.strip()) for x in args.ivfflat_probes.split(",") if x.strip()]

    queries = DEFAULT_CLINICAL_QUERIES
    if limit_queries is not None and limit_queries > 0:
        queries = queries[:limit_queries]

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = UnifiedBenchmarkRunner()
    report = runner.run_full_suite(
        queries=queries,
        top_k=args.top_k,
        warmup_runs=warmup,
        iterations=iterations,
        measure_cold=measure_cold,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search_values=ef_search_vals,
        ivfflat_lists=args.ivfflat_lists,
        ivfflat_probes_values=probes_vals,
        include_hybrid=not args.no_hybrid
    )

    # Export structured JSON
    json_path = out_dir / "benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Saved structured benchmark report -> {json_path.name}")

    # Export raw queries CSV
    raw_csv_path = out_dir / "benchmark_raw_queries.csv"
    export_raw_queries_csv(report["raw_samples"], raw_csv_path, report["telemetry"])
    print(f"[+] Saved raw query measurements ({len(report['raw_samples'])} samples) -> {raw_csv_path.name}")

    # Export summary CSV
    summary_csv_path = out_dir / "benchmark_summary.csv"
    export_summary_csv(report["summary"], summary_csv_path, report["telemetry"])
    print(f"[+] Saved aggregated summary table -> {summary_csv_path.name}")

    # Export Markdown summary
    md_summary = format_markdown_summary(report)
    md_path = out_dir / "benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)
    print(f"[+] Saved markdown comparison table -> {md_path.name}")

    # Print summary to console
    print("\n" + md_summary + "\n")

    # Optional Plotting
    if args.plot:
        try:
            from scripts.plot_benchmarks import plot_benchmark_charts
            plot_benchmark_charts(out_dir)
        except Exception as e:
            print(f"[!] Plot generation notice: {e}")

if __name__ == "__main__":
    main()
