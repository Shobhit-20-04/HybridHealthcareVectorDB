# Benchmark Evaluation Summary Report

**Hardware:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (16 cores) | 15.69 GB RAM | Windows-10-10.0.26300-SP0  
**Database:** PostgreSQL 18.6 | `pgvector` v0.8.6  
**Model & Space:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, Metric: `cosine` `<=>`)  
**Parameters:** Top-K = 10 | Warmup = 0 | Iterations = 1 | Total Duration = 6.15s

## Indexing Strategy & Query Performance Comparison

| Configuration | Index Type | Build Time | Size | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Cold P50 | QPS | 1-Recall | ID-Recall | Noise Status |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **Exact Flat (Ground Truth Baseline)** | Flat | - | 0 bytes | 18.904 | 18.712 | 19.095 | 19.095 | 19.095 | - | 52.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **HNSW (m=16, ef_s=40)** | HNSW | 612.3 ms | 12 MB | 20.889 | 20.733 | 21.044 | 21.044 | 21.044 | - | 47.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **IVFFlat (lists=25, probes=5)** | IVFFlat | 385.3 ms | 10128 kB | 20.460 | 20.092 | 20.829 | 20.829 | 20.829 | - | 48.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **Exact Flat [Low Selectivity (Broad Filter ~70%)]** | exact_flat | - | - | 257.486 | 203.682 | 311.290 | 311.290 | 311.290 | - | 3.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **HNSW Planner Choice [Low Selectivity (Broad Filter ~70%)]** | planner_choice | - | - | 205.920 | 202.342 | 209.498 | 209.498 | 209.498 | - | 4.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **Exact Flat [Moderate Selectivity (~20%)]** | exact_flat | - | - | 60.527 | 59.376 | 61.678 | 61.678 | 61.678 | - | 16.5 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **HNSW Planner Choice [Moderate Selectivity (~20%)]** | planner_choice | - | - | 112.410 | 59.258 | 165.563 | 165.563 | 165.563 | - | 8.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE; HIGH_VARIANCE` |
| **Exact Flat [High Selectivity / Restrictive (~1.5%)]** | exact_flat | - | - | 113.856 | 62.290 | 165.422 | 165.422 | 165.422 | - | 8.8 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE; HIGH_VARIANCE` |
| **HNSW Planner Choice [High Selectivity / Restrictive (~1.5%)]** | planner_choice | - | - | 54.052 | 52.215 | 55.888 | 55.888 | 55.888 | - | 18.5 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE` |
| **Exact Flat [Zero-Match / Ultra-Restrictive (0%)]** | exact_flat | - | - | 156.394 | 99.916 | 212.872 | 212.872 | 212.872 | - | 6.4 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE; HIGH_VARIANCE` |
| **HNSW Planner Choice [Zero-Match / Ultra-Restrictive (0%)]** | planner_choice | - | - | 145.499 | 83.143 | 207.854 | 207.854 | 207.854 | - | 6.9 | 100.00% | 100.00% | `SMALL_SAMPLE_SIZE; HIGH_VARIANCE` |