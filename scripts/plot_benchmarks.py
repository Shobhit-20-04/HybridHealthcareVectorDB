"""
Benchmark Visualization Script (Pure-Python SVG & Optional Matplotlib)
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Renders high-resolution vector SVG comparison charts directly from results/benchmark_report.json:
1. benchmark_latency_recall.svg: Mean query latency (ms) vs. Recall@10 accuracy
2. benchmark_percentiles.svg: Latency distribution (Min, P50, P90, P95, Max)
3. benchmark_hybrid_selectivity.svg: Query latency across selectivity tiers (Low, Moderate, High, Zero)
4. benchmark_cold_vs_warm.svg: Cold cache vs. warm cache P50 latency comparison
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def render_svg_barchart(
    title: str,
    categories: List[str],
    series_a: List[float],
    series_a_label: str,
    series_b: Optional[List[float]] = None,
    series_b_label: Optional[str] = None,
    width: int = 900,
    height: int = 450,
    y_max_a: Optional[float] = None,
    y_max_b: Optional[float] = None
) -> str:
    """Renders a pure-Python dual-axis or single-axis SVG bar/line chart."""
    margin_top = 60
    margin_bottom = 90
    margin_left = 80
    margin_right = 80 if series_b else 30

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_a = y_max_a or (max(series_a) * 1.15 if series_a and max(series_a) > 0 else 1.0)
    max_b = y_max_b or (max(series_b) * 1.15 if series_b and max(series_b) > 0 else 1.0)

    n_bars = len(categories)
    bar_group_w = plot_w / max(1, n_bars)
    bar_w = min(40, bar_group_w * 0.5)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif; background: #ffffff;">')

    # Chart Title
    svg.append(f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">{title}</text>')

    # Background grid lines & Y-axis labels for series A
    n_ticks = 5
    for i in range(n_ticks + 1):
        y_frac = i / n_ticks
        y_pos = margin_top + plot_h - (y_frac * plot_h)
        val_a = y_frac * max_a

        # Grid line
        svg.append(f'<line x1="{margin_left}" y1="{y_pos:.1f}" x2="{margin_left + plot_w}" y2="{y_pos:.1f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,4"/>')
        # Left Y label
        svg.append(f'<text x="{margin_left - 12}" y="{y_pos + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{val_a:.2f}</text>')

        # Right Y label (if dual axis)
        if series_b:
            val_b = y_frac * max_b
            pct_str = f"{val_b * 100:.0f}%" if max_b <= 1.05 else f"{val_b:.1f}"
            svg.append(f'<text x="{margin_left + plot_w + 12}" y="{y_pos + 4:.1f}" text-anchor="start" font-size="11" fill="#16a34a">{pct_str}</text>')

    # Axis Lines
    svg.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>')
    if series_b:
        svg.append(f'<line x1="{margin_left + plot_w}" y1="{margin_top}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#16a34a" stroke-width="1.5"/>')

    # Axis Title Labels
    svg.append(f'<text transform="rotate(-90)" x="{- (margin_top + plot_h / 2)}" y="{margin_left - 48}" text-anchor="middle" font-size="12" font-weight="bold" fill="#2563eb">{series_a_label}</text>')
    if series_b:
        svg.append(f'<text transform="rotate(90)" x="{margin_top + plot_h / 2}" y="{- (margin_left + plot_w + 48)}" text-anchor="middle" font-size="12" font-weight="bold" fill="#16a34a">{series_b_label}</text>')

    # Plot Bars (Series A) and Overlay Points/Lines (Series B)
    line_points = []
    for idx, (cat, val_a) in enumerate(zip(categories, series_a)):
        center_x = margin_left + idx * bar_group_w + (bar_group_w / 2)
        bar_x = center_x - (bar_w / 2)
        bar_h = (val_a / max_a) * plot_h if max_a > 0 else 0
        bar_y = margin_top + plot_h - bar_h

        # Bar
        svg.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#3b82f6" rx="3" ry="3"/>')
        # Value label atop bar
        svg.append(f'<text x="{center_x:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e3a8a">{val_a:.2f}</text>')

        # Category label under X-axis
        # Rotate if category string is long
        svg.append(f'<text transform="rotate(25, {center_x:.1f}, {margin_top + plot_h + 18})" x="{center_x:.1f}" y="{margin_top + plot_h + 18}" text-anchor="start" font-size="10" fill="#334155">{cat}</text>')

        # Series B point
        if series_b and idx < len(series_b):
            val_b = series_b[idx]
            pt_y = margin_top + plot_h - ((val_b / max_b) * plot_h if max_b > 0 else 0)
            line_points.append((center_x, pt_y, val_b))

    # Draw Series B connecting line & markers
    if series_b and len(line_points) > 1:
        polyline_coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in line_points)
        svg.append(f'<polyline fill="none" stroke="#16a34a" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline_coords}"/>')
        for x, y, vb in line_points:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#16a34a" stroke="#ffffff" stroke-width="2"/>')
            svg.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d">{vb * 100:.1f}%</text>')

    # Legend
    legend_y = height - 15
    svg.append(f'<rect x="{margin_left + 20}" y="{legend_y - 10}" width="14" height="14" fill="#3b82f6" rx="2"/>')
    svg.append(f'<text x="{margin_left + 40}" y="{legend_y + 2}" font-size="11" fill="#334155">{series_a_label}</text>')
    if series_b:
        svg.append(f'<line x1="{margin_left + 220}" y1="{legend_y - 3}" x2="{margin_left + 245}" y2="{legend_y - 3}" stroke="#16a34a" stroke-width="3"/>')
        svg.append(f'<circle cx="{margin_left + 232}" cy="{legend_y - 3}" r="4" fill="#16a34a"/>')
        svg.append(f'<text x="{margin_left + 255}" y="{legend_y + 2}" font-size="11" fill="#334155">{series_b_label}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

def plot_benchmark_charts(results_dir: Path):
    """Parses benchmark_report.json and renders vector charts into results directory."""
    json_path = results_dir / "benchmark_report.json"
    if not json_path.exists():
        print(f"[!] Cannot generate charts: Missing {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", [])
    if not summary:
        print("[!] No summary records in report to chart.")
        return

    # Filter Pure Vector Index entries
    vector_entries = [r for r in summary if r.get("category") == "pure_vector"]
    hybrid_entries = [r for r in summary if r.get("category") == "hybrid_selectivity"]

    # -------------------------------------------------------------------------
    # Chart 1: Latency vs. Recall@10
    # -------------------------------------------------------------------------
    if vector_entries:
        cat_labels = [r["configuration"].split("(")[0].strip() + (f" ({r['configuration'].split('(')[1]}" if "(" in r["configuration"] else "") for r in vector_entries]
        mean_lats = [r["warm_latency"]["mean"] for r in vector_entries]
        recalls = [r["distance_recall"] for r in vector_entries]

        svg_content = render_svg_barchart(
            title="Vector Index Performance: Mean Latency (ms) vs. Recall@10",
            categories=cat_labels,
            series_a=mean_lats,
            series_a_label="Mean Latency (ms)",
            series_b=recalls,
            series_b_label="Recall@10",
            y_max_b=1.05
        )
        out_svg1 = results_dir / "benchmark_latency_recall.svg"
        with open(out_svg1, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"[+] Rendered vector chart -> {out_svg1.name}")

    # -------------------------------------------------------------------------
    # Chart 2: Latency Percentiles (P50 vs P95 vs P99)
    # -------------------------------------------------------------------------
    if vector_entries:
        cat_labels = [r["configuration"].split("(")[0].strip() for r in vector_entries]
        p50_lats = [r["warm_latency"]["p50"] for r in vector_entries]
        p95_lats = [r["warm_latency"]["p95"] for r in vector_entries]

        svg_content = render_svg_barchart(
            title="Vector Search Latency Percentiles (P50 vs. P95)",
            categories=cat_labels,
            series_a=p50_lats,
            series_a_label="P50 Latency (ms)",
            series_b=p95_lats,
            series_b_label="P95 Latency (ms)"
        )
        out_svg2 = results_dir / "benchmark_percentiles.svg"
        with open(out_svg2, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"[+] Rendered vector chart -> {out_svg2.name}")

    # -------------------------------------------------------------------------
    # Chart 3: Hybrid Selectivity Tiers
    # -------------------------------------------------------------------------
    if hybrid_entries:
        cat_labels = [f"{r['configuration'].replace('Exact Flat [', '').replace('HNSW Planner Choice [', '').replace(']', '')} ({r.get('selectivity_percentage', 0)}%)" for r in hybrid_entries]
        hyb_lats = [r["warm_latency"]["mean"] for r in hybrid_entries]

        svg_content = render_svg_barchart(
            title="Hybrid Relational-Vector Query Latency Across Selectivity Tiers",
            categories=cat_labels,
            series_a=hyb_lats,
            series_a_label="Query Latency (ms)"
        )
        out_svg3 = results_dir / "benchmark_hybrid_selectivity.svg"
        with open(out_svg3, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"[+] Rendered vector chart -> {out_svg3.name}")

    # Optional Matplotlib PNG generation if installed
    try:
        import matplotlib.pyplot as plt
        # Render PNG version for backwards compatibility
        fig, ax = plt.subplots(figsize=(10, 5))
        if vector_entries:
            names = [r["configuration"] for r in vector_entries]
            lats = [r["warm_latency"]["mean"] for r in vector_entries]
            ax.bar(names, lats, color="#3b82f6")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Vector Index Mean Latency")
            plt.xticks(rotation=30, ha="right")
            png_out = results_dir / "benchmark_charts.png"
            plt.savefig(png_out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[+] Rendered PNG chart -> {png_out.name}")
    except ImportError:
        pass

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    res_dir = base_dir / "results"
    plot_benchmark_charts(res_dir)
