# Benchmark Evaluation Summary Report

**Hardware:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (16 cores) | 15.69 GB RAM | Windows-10-10.0.26300-SP0  
**Database:** PostgreSQL 18.6 | `pgvector` v0.8.6  
**Model & Space:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, Metric: `cosine` `<=>`)  
**Parameters:** Top-K = 10 | Warmup = 1 | Iterations = 3 | Total Duration = 43.01s

## Indexing Strategy & Query Performance Comparison

| Configuration | Index Type | Build Time | Size | Mean (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Cold P50 | QPS | 1-Recall | ID-Recall | Noise Status |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| **Exact Flat (Ground Truth Baseline)** | Flat | - | 0 bytes | 10.797 | 10.799 | 11.191 | 11.526 | 11.988 | 10.552 | 92.6 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=10)** | HNSW | 55.4 ms | 184 kB | 11.256 | 11.189 | 11.708 | 11.937 | 12.000 | 11.341 | 88.8 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=40)** | HNSW | 55.4 ms | 184 kB | 11.531 | 11.384 | 12.205 | 12.354 | 13.554 | 11.394 | 86.7 | 100.00% | 100.00% | `STABLE` |
| **HNSW (m=16, ef_s=100)** | HNSW | 55.4 ms | 184 kB | 11.225 | 11.215 | 11.516 | 11.682 | 12.331 | 11.387 | 89.1 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=1)** | IVFFlat | 169.2 ms | 1472 kB | 11.352 | 11.272 | 11.785 | 11.844 | 13.409 | 11.478 | 88.1 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=5)** | IVFFlat | 169.2 ms | 1472 kB | 11.204 | 11.141 | 11.574 | 11.609 | 12.347 | 11.228 | 89.2 | 100.00% | 100.00% | `STABLE` |
| **IVFFlat (lists=25, probes=10)** | IVFFlat | 169.2 ms | 1472 kB | 11.224 | 11.256 | 11.585 | 11.614 | 12.165 | 11.41 | 89.1 | 100.00% | 100.00% | `STABLE` |
| **Exact Flat [Low Selectivity (Broad Filter ~70%)]** | exact_flat | - | - | 103.091 | 59.146 | 168.251 | 168.896 | 168.896 | - | 9.7 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [Low Selectivity (Broad Filter ~70%)]** | planner_choice | - | - | 68.197 | 52.574 | 155.986 | 166.219 | 166.219 | - | 14.7 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [Moderate Selectivity (~20%)]** | exact_flat | - | - | 57.275 | 49.777 | 57.355 | 151.606 | 151.606 | - | 17.5 | 100.00% | 100.00% | `STABLE` |
| **HNSW Planner Choice [Moderate Selectivity (~20%)]** | planner_choice | - | - | 77.588 | 51.130 | 153.409 | 156.112 | 156.112 | - | 12.9 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [High Selectivity / Restrictive (~1.5%)]** | exact_flat | - | - | 69.930 | 49.944 | 153.153 | 154.207 | 154.207 | - | 14.3 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [High Selectivity / Restrictive (~1.5%)]** | planner_choice | - | - | 71.521 | 50.224 | 160.200 | 163.698 | 163.698 | - | 14.0 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **Exact Flat [Zero-Match / Ultra-Restrictive (0%)]** | exact_flat | - | - | 61.147 | 46.704 | 148.377 | 150.399 | 150.399 | - | 16.4 | 100.00% | 100.00% | `HIGH_VARIANCE` |
| **HNSW Planner Choice [Zero-Match / Ultra-Restrictive (0%)]** | planner_choice | - | - | 57.062 | 49.487 | 52.814 | 162.220 | 162.220 | - | 17.5 | 100.00% | 100.00% | `HIGH_VARIANCE` |