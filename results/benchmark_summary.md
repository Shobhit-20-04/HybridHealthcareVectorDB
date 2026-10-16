# Benchmark Evaluation Summary Report

**Hardware:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (16 cores) | 15.69 GB RAM | Windows-10-10.0.26300-SP0  
**Database:** PostgreSQL 18.6 | `pgvector` v0.8.6  
**Model & Space:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, Metric: `cosine` `<=>`)  
**Parameters:** Top-K = 10 | Warmup = 1 | Iterations = 3 | Total Duration = 43.26s

## Indexing Strategy & Query Performance Comparison

| Configuration | Index Type | Build Time | Size | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Cold P50 | QPS | 1-Recall | ID-Recall | Noise Status |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **Exact Flat (Ground Truth Baseline)** | Flat | - | 0 bytes | 10.649 | 10.649 | 10.976 | 11.234 | 11.391 | 10.389 | 93.9 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=10)** | HNSW | 59.3 ms | 184 kB | 11.480 | 11.276 | 11.867 | 12.336 | 18.175 | 11.11 | 87.1 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=40)** | HNSW | 59.3 ms | 184 kB | 11.392 | 11.300 | 11.831 | 11.953 | 14.890 | 11.296 | 87.8 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=100)** | HNSW | 59.3 ms | 184 kB | 11.329 | 11.289 | 11.736 | 11.869 | 12.670 | 11.342 | 88.3 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=1)** | IVFFlat | 47.2 ms | 1472 kB | 11.284 | 11.180 | 11.684 | 11.724 | 16.264 | 11.297 | 88.6 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=5)** | IVFFlat | 47.2 ms | 1472 kB | 11.259 | 11.231 | 11.582 | 11.632 | 11.754 | 11.234 | 88.8 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=10)** | IVFFlat | 47.2 ms | 1472 kB | 11.292 | 11.283 | 11.662 | 11.819 | 11.976 | 11.234 | 88.6 | 100.00% | 100.00% | `STABLE` |
| **Exact Flat [Low Selectivity (Broad Filter ~70%)]** | exact_flat | - | - | 80.653 | 51.969 | 157.663 | 165.122 | 165.122 | - | 12.4 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [Low Selectivity (Broad Filter ~70%)]** | planner_choice | - | - | 82.696 | 53.094 | 167.181 | 168.136 | 168.136 | - | 12.1 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [Moderate Selectivity (~20%)]** | exact_flat | - | - | 64.190 | 50.701 | 150.081 | 150.629 | 150.629 | - | 15.6 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [Moderate Selectivity (~20%)]** | planner_choice | - | - | 64.540 | 49.283 | 148.488 | 167.735 | 167.735 | - | 15.5 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [High Selectivity / Restrictive (~1.5%)]** | exact_flat | - | - | 77.140 | 49.297 | 158.124 | 160.949 | 160.949 | - | 13.0 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [High Selectivity / Restrictive (~1.5%)]** | planner_choice | - | - | 77.712 | 50.006 | 155.222 | 162.341 | 162.341 | - | 12.9 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [Zero-Match / Ultra-Restrictive (0%)]** | exact_flat | - | - | 75.718 | 47.705 | 155.527 | 155.980 | 155.980 | - | 13.2 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [Zero-Match / Ultra-Restrictive (0%)]** | planner_choice | - | - | 69.417 | 48.073 | 152.321 | 169.320 | 169.320 | - | 14.4 | 100.00% | 100.00% | `HIGH_VARIANCE` |