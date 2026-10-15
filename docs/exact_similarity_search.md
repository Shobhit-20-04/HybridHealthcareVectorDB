# Exact Semantic Retrieval Engine (Ground-Truth Flat Search Baseline)
**Project:** Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval  
**Scope:** Academic DBMS Architecture & Experiments (Zero External APIs, Headless CLI Only)

---

## 1. Overview & Purpose

In vector database research and information retrieval, approximate nearest neighbor (ANN) indexes—such as **IVFFlat** (inverted file partitioning) and **HNSW** (Hierarchical Navigable Small World graphs)—trade off retrieval accuracy (Recall@K) in exchange for sub-millisecond query latency.

To rigorously benchmark ANN recall and latency trade-offs without synthetic or unsubstantiated claims, an academic DBMS architecture requires an **authoritative ground-truth baseline**:
- **Exact Sequential Scan (Flat Search):** Evaluates the exact brute-force cosine distance across every single vector in the corpus.
- **100% Recall Anchor:** Guarantees that the true $k$ nearest semantic neighbors are always identified.
- **Index-Independent Execution:** Forces the PostgreSQL cost-based query optimizer to bypass approximate indexes, executing pure sequential disk and buffer scans.

---

## 2. Mathematical Formulations: Cosine Distance vs. Cosine Similarity

A common point of ambiguity in vector database literature is the distinction between **Cosine Similarity** and **Cosine Distance**. The engine explicitly defines and reports both metrics:

### 2.1. Cosine Similarity ($S_C$)
Measures the cosine of the angle $\theta$ between query vector $u$ and document vector $v$:
$$S_C(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- **Domain Range:** $S_C \in [-1.0, 1.0]$.
- **Interpretation:** **Higher is closer.**
  - $S_C = 1.0$: Identical vectors (collinear, pointing in identical directions).
  - $S_C = 0.0$: Orthogonal vectors (statistically uncorrelated in semantic space).
  - $S_C = -1.0$: Diametrically opposite vectors.

### 2.2. Cosine Distance ($D_C$)
Used by PostgreSQL and `pgvector` via the `<=>` operator:
$$D_C(u, v) = 1.0 - S_C(u, v) = 1.0 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

- **Domain Range:** $D_C \in [0.0, 2.0]$.
- **Interpretation:** **Lower is closer.**
  - $D_C = 0.0$: Identical vectors ($\theta = 0^\circ$).
  - $D_C = 1.0$: Orthogonal vectors ($\theta = 90^\circ$).
  - $D_C = 2.0$: Diametrically opposite vectors ($\theta = 180^\circ$).
- **Ordering Property:** Ascending sort (`ORDER BY embedding <=> query_vector ASC`) correctly ranks the most semantically relevant documents at Rank #1.

---

## 3. PostgreSQL Query Architecture & Execution Plan

### 3.1. Disabling Approximate Index Scans
To ensure that PostgreSQL executes a pure Flat Sequential Scan rather than traversing an HNSW graph or IVFFlat Voronoi centroid list, the engine wraps queries in a session setting:
```sql
BEGIN;
SET LOCAL enable_indexscan = off;
SET LOCAL enable_bitmapscan = off;

SELECT 
    ce.chunk_id,
    cd.title,
    dc.chunk_text,
    (ce.embedding <=> %s::vector) AS cosine_distance,
    (1.0 - (ce.embedding <=> %s::vector)) AS cosine_similarity
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> %s::vector ASC
LIMIT 5;

COMMIT;
```

### 3.2. Physical Query Plan (`EXPLAIN ANALYZE, BUFFERS`)
```text
Limit  (cost=29.13..29.14 rows=3 width=32) (actual time=0.475..0.476 rows=3 loops=1)
  Buffers: shared hit=29
  ->  Sort  (cost=29.13..29.39 rows=103 width=32) (actual time=0.474..0.474 rows=3 loops=1)
        Sort Key: ((embedding <=> '[...]'::vector))
        Sort Method: top-N heapsort  Memory: 25kB
        Buffers: shared hit=29
        ->  Seq Scan on chunk_embeddings ce  (cost=0.00..27.80 rows=103 width=32) (actual time=0.013..0.101 rows=97 loops=1)
              Filter: ((model_id)::text = 'sentence-transformers/all-MiniLM-L6-v2'::text)
              Buffers: shared hit=26
Planning Time: 3.249 ms
Execution Time: 0.565 ms
```
The plan demonstrates that:
1. `Seq Scan on chunk_embeddings` scans all candidate vectors sequentially.
2. `top-N heapsort` maintains a bounded priority queue of size $k$ in memory without allocating full sort memory.
3. Zero index structures were accessed.

---

## 4. CLI Execution Guide

### 4.1. Run Sample Semantic Queries
```powershell
# Standard clinical query (Top-5 results)
.\.venv\Scripts\python.exe scripts/search_exact.py "progressive dyspnea on exertion and lower extremity edema"

# Custom Top-K and physical EXPLAIN plan inspection
.\.venv\Scripts\python.exe scripts/search_exact.py "severe pleuritic chest pain and fever" --top-k 3 --explain

# Structured JSON output for downstream analysis
.\.venv\Scripts\python.exe scripts/search_exact.py "diabetic ketoacidosis and elevated anion gap" --json --top-k 2
```

---

## 5. Automated Test Suite

Run the dedicated exact retrieval test suite:
```powershell
.\.venv\Scripts\pytest.exe tests/test_exact_retrieval.py -v
```

Tests verify:
- Strict ascending distance ranking order ($D_{C, 1} \le D_{C, 2} \le \dots \le D_{C, k}$).
- Mathematical invariance: $S_C + D_C = 1.0 \pm 10^{-5}$ across all returned matches.
- Input validation: empty or whitespace-only queries reject with `ValueError`.
- Boundary validation: $k \le 0$ rejects with `ValueError`.
- Error handling: unregistered models or missing embeddings reject cleanly.
- Model dimension mismatch: verified against PostgreSQL catalog.
- Deterministic reproducibility: duplicate calls yield bit-identical distances and ordering.
- Execution plan verification: verified that `Seq Scan` is used and `Index Scan` is absent.
