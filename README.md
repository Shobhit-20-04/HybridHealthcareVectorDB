# Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

[![Database: PostgreSQL + pgvector](https://img.shields.io/badge/Database-PostgreSQL%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic database management system (DBMS) project investigating unified relational and vector database architectures for clinical information retrieval. This research focuses strictly on database internals, indexing tradeoffs, hybrid relational-plus-vector query optimization, access control, and empirical performance benchmarking—**not** on end-user web applications, dashboards, hospital portals, or conversational chatbots.

---

## 1. Project Motivation & Scope

In contemporary healthcare informatics, clinical data exists in two primary modalities:
1. **Structured Data:** Patient demographics, encounters, ICD-10 diagnostic codes, laboratory panels, and medication histories requiring relational integrity, foreign key constraints, and ACID guarantees.
2. **Unstructured & Semistructured Data:** Free-text clinical discharge summaries, progress notes, pathology evaluations, and radiological imaging requiring dense high-dimensional semantic vector search.

Dual-database architectures (pairing a relational database with a standalone vector database) suffer from synchronization lag, two-phase network query penalties, and duplicated access-control mechanisms. 

This project explores a **unified hybrid architecture** using **PostgreSQL** paired with the **`pgvector`** extension to evaluate:
- Single-transaction hybrid queries combining relational SQL predicates (e.g., date ranges, clinical departments) with dense vector distance calculations.
- Comparative index performance: Exact Sequential Scan (brute-force Flat) vs. Inverted File Flat (IVFFlat) vs. Hierarchical Navigable Small World (HNSW) graphs.
- Multimodal embedding integration (clinical text and diagnostic imaging representations).
- Native database-level security and Row-Level Security (RLS) enforcement.
- Empirical benchmarking of query latency ($P_{50}, P_{95}, P_{99}$), Throughput (QPS), Recall@K, and index footprint across varying relational filter selectivities.

---

## 2. Repository Layout

```
D:\HybridHealthcareVectorDB\
├── .env.example              # Template environment configuration (credentials, ports, models)
├── .gitignore                # Git exclusions (credentials, caches, artifacts, datasets)
├── requirements.txt          # Python dependencies (drivers, transformers, analysis, testing)
├── README.md                 # Project documentation and architectural overview
├── database/                 # Database initialization, migration scripts, and connection pooler
├── sql/                      # PostgreSQL DDL, indexing, hybrid query, and RLS scripts
│   ├── 01_extensions.sql     # Extension initialization (vector, uuid-ossp, pgcrypto)
│   ├── 02_schema.sql         # Relational tables, constraints, and vector column definitions
│   ├── 03_indexes.sql        # B-tree, JSONB GIN, and pgvector HNSW/IVFFlat index creation
│   └── 04_seed_embedding_models.sql # Explicit model registry seeding (384-d, 512-d, 768-d)
├── src/                      # Source modules for embedding, retrieval, security, and benchmarking
│   ├── embeddings/           # Text (sentence-transformers) and multimodal (CLIP) pipeline
│   ├── retrieval/            # Query execution engines (exact, IVFFlat, HNSW, hybrid)
│   ├── benchmark/            # Benchmarking harness (latency, QPS, Recall@K, memory)
│   └── security/             # Role-based access control and session management
├── data/                     # Data tiers (strictly segregated, raw/processed ignored by git)
│   ├── raw/                  # Source clinical datasets or raw files
│   ├── processed/            # Ingestion-ready tabular and text files
│   └── synthetic/            # Synthetic healthcare record generator fixtures
├── tests/                    # Automated unit, integration, and database test suites
├── results/                  # Empirical benchmark metrics, JSON outputs, and plotted charts
├── docs/                     # Architectural documentation, specs, and implementation plans
│   ├── implementation_plan.md# Detailed phased development roadmap
│   ├── schema_design.md      # Data dictionary, Mermaid ER diagram, and normalization rationale
│   └── database_setup_guide.md# Windows prerequisite checks and pgvector setup instructions
└── scripts/                  # Standalone administrative and execution scripts
    ├── validate_sql.py       # Offline DDL and referential integrity validator
    └── init_database.py      # Automated database migration runner
```

---

## 3. Architecture Modules Overview

| Module | Core Functionality | Database / Technical Implementation |
|---|---|---|
| **Relational Schema** | Normalized representation of patients, encounters, diagnoses, notes | PostgreSQL DDL with UUID primary keys, foreign key constraints, B-Tree indexes |
| **Document Storage** | Storage of free-text notes and semistructured metadata | `TEXT` clinical narrative storage paired with `JSONB` metadata columns |
| **Vector Embeddings** | Generation of dense vector representations for notes and imaging | HuggingFace `sentence-transformers` (384/768-dim) and `CLIP/BiomedCLIP` (512-dim) |
| **Exact Similarity Search** | Gold-standard baseline nearest-neighbor retrieval | Sequential scan with L2 (`<->`), Cosine (`<=>`), or Inner Product (`<#>`) |
| **IVFFlat Indexing** | Inverted file partitioning into Voronoi cells | `USING ivfflat (embedding vector_cosine_ops) WITH (lists = ...)` |
| **HNSW Indexing** | Multi-layer proximity graph search | `USING hnsw (embedding vector_cosine_ops) WITH (m = ..., ef_construction = ...)` |
| **Hybrid SQL Queries** | Combined relational filtering + semantic distance ranking | Pre-filtered attribute constraints vs. post-filtered vector retrieval |
| **Multimodal Storage** | Joint storage and retrieval of imaging metadata and visual embeddings | Unified schema storing radiological metadata + dense image vectors |
| **Access Control (RLS)** | Least-privilege healthcare data protection | PostgreSQL `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` with role policies |
| **Benchmarking Suite** | Quantitative profiling across indexing methods and scales | Empirical measurement of Recall@K, Latency ($P_{50}/P_{95}/P_{99}$), QPS, Disk Footprint |
| **CLI RAG Demo** | Headless terminal verification of hybrid retrieval grounding | Interactive terminal-based semantic retrieval and synthesis pipeline |

---

## 4. Prerequisites & Installation

### 4.1 Prerequisites
- **Operating System:** Windows 10 / 11 (x64)
- **Python:** 3.11.x (installed and accessible in PATH)
- **PostgreSQL:** Version 16+ (PostgreSQL 18.x installed locally on system)
- **Database Extension:** `pgvector`

### 4.2 Installing `pgvector` on Windows
`pgvector` is a C extension for PostgreSQL that enables vector storage and indexing.

#### Option A: Precompiled Extension Binaries (Recommended for Windows)
1. Obtain the precompiled `pgvector` binaries compatible with your PostgreSQL version.
2. Copy `vector.dll` into:
   ```
   C:\Program Files\PostgreSQL\18\lib\
   ```
3. Copy `vector.control` and all `vector--*.sql` files into:
   ```
   C:\Program Files\PostgreSQL\18\share\extension\
   ```
4. Restart the PostgreSQL service:
   ```powershell
   Restart-Service postgresql-x64-18
   ```

#### Option B: Building from Source via MSVC
1. Open the **x64 Native Tools Command Prompt for VS**.
2. Set the PostgreSQL root directory:
   ```cmd
   set "PGROOT=C:\Program Files\PostgreSQL\18"
   ```
3. Clone and build `pgvector`:
   ```cmd
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /F Makefile.msvc
   nmake /F Makefile.msvc install
   ```

#### Option C: Enabling the Extension in PostgreSQL
Connect to your database via `psql` and execute:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

---

## 5. Environment & Credentials Configuration

This project strictly follows credential-safety guidelines. No passwords or sensitive connection strings are hardcoded into source files.

1. Copy the example configuration template:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Edit `.env` with your local PostgreSQL credentials:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=healthcare_vectordb
   DB_USER=postgres
   DB_PASSWORD=your_secure_password
   ```

---

## 6. Python Environment Setup

1. Create and activate a Python 3.11 virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Upgrade `pip` and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. Academic Research & Benchmarking Rigor

In strict compliance with academic standards:
- **No Synthesized Citations:** All architectural models, algorithms, and metric definitions cite established literature directly when formally documented.
- **Empirical Measurement Only:** All benchmark figures (QPS, recall, millisecond latencies) published in `results/` are measured and output by executing the benchmarking scripts on actual database instances; no synthetic performance numbers are fabricated.
- **Reproducibility:** All schema scripts, index configurations, synthetic data generators, and benchmarking runs are fully automated and version-controlled.

---

## 8. Development Roadmap

- [x] **Phase 1:** Project foundation, directory architecture, environment configuration, and implementation specification.
- [x] **Phase 2:** PostgreSQL schema design, DDL definitions, pgvector indexing, and setup guides.
- [x] **Phase 3:** Synthetic healthcare dataset generator (`src/synthetic_generator.py`).
- [x] **Phase 4:** Dense embedding generation and batch ingestion pipeline (`src/embeddings/`).
- [x] **Phase 5:** Indexing implementation (Exact, IVFFlat, HNSW) and hybrid query engine (`src/retrieval/`).
- [x] **Phase 6:** Row-Level Security (RLS) policies and role-based access testing (`src/security/`, `sql/06_security_rls.sql`).
- [x] **Phase 7:** Rigorous experimental benchmark suite execution and result visualization (`src/benchmark/`, `results/`).
- [x] **Phase 8:** Command-line RAG demonstration (`src/retrieval/cli_rag.py`) and automated test suite (`tests/`).

---

## 9. Empirical Experimental Results

All benchmark metrics are directly measured on synthetic clinical datasets ($N = 453$ passages, 384 dimensions) using [`src/benchmark/benchmark_engine.py`](file:///D:/HybridHealthcareVectorDB/src/benchmark/benchmark_engine.py).

### Table 1: Indexing Strategy Trade-offs (Corpus Size: 453 vectors)
| Strategy | Build Time (ms) | Recall@10 | Mean Latency (ms) | P50 (ms) | P95 (ms) | Throughput (QPS) |
|---|---|---|---|---|---|---|
| **Exact Flat (Baseline)** | 0.00 | 1.000 | 1.483 | 1.462 | 1.667 | 674.2 |
| **IVFFlat (lists=10, probes=2)** | 1864.48 | 1.000 | 0.551 | 0.470 | 0.878 | 1814.6 |
| **HNSW (ef_search=40)** | 10.56 | 0.076 | 0.215 | 0.213 | 0.232 | 4642.1 |

### Table 2: Hybrid Query Strategies Execution Times
| Strategy | Relational Selectivity | Mean Latency (ms) | P50 (ms) | P95 (ms) | Throughput (QPS) |
|---|---|---|---|---|---|
| **Pure Vector Search** | Unfiltered (100% rows) | 1.631 | 1.418 | 2.177 | 613.0 |
| **Pre-Filtering (Attribute-First)** | High Selectivity (~15% matches) | **0.907** | **0.895** | **0.993** | **1102.9** |
| **Post-Filtering (Vector-First)** | High Selectivity (~15% matches) | 5.422 | 5.356 | 5.943 | 184.4 |

### Architectural Insights & Findings
1. **Pre-Filtering Superiority on Selective Predicates:**
   - Under narrow clinical cohort filtering (e.g. `department = 'Cardiology'`), Pre-Filtering achieves **0.907 ms** versus **5.422 ms** for Post-Filtering (**~6x throughput speedup**). Pre-filtering discards non-matching rows using relational B-Tree indexes before performing high-dimensional vector distance calculations.
2. **IVFFlat vs. Flat Search Trade-off:**
   - IVFFlat clustered vector space into 10 Voronoi centroids, achieving **1.0 Recall@10** while reducing query latency by **62.8%** (0.551 ms vs 1.483 ms).
3. **Database-Managed Access Control (RLS):**
   - By enforcing Row-Level Security at the database kernel level (`ALTER TABLE ... ENABLE ROW LEVEL SECURITY`), clinical role isolation is maintained directly inside the query execution plan, eliminating application-layer permission leakage risks.

---

## 10. Ingestion Pipeline & Execution Commands

```powershell
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Generate & Sanitize Deterministic Synthetic Healthcare Data
python scripts/generate_data.py --preset small --seed 42

# 3. Transactionally Load Data into PostgreSQL (with Check Constraints & FK Validation)
python scripts/load_data.py --truncate

# 4. Generate Dense Vector Embeddings & Store in pgvector (384-d, Cosine)
python scripts/embed_documents.py --smoke-test

# 5. Execute Exact Semantic Retrieval (Ground-Truth Flat Scan Baseline)
python scripts/search_exact.py "progressive dyspnea on exertion" --top-k 5 --explain

# 6. Audit Live Table Counts and Manifest Synchronization
python scripts/count_records.py

# 7. Run Database Referential & Invariant Integrity Audits
python scripts/validate_ingestion.py

# 8. Run Complete Automated Test Suite (106 Unit, Ingestion, Retrieval, Multimodal, Security, Benchmark & RAG Tests)
pytest tests/ -v

# 9. Execute Master Unified Reproducible Database Benchmark Suite (Flat vs HNSW vs IVFFlat + 4 Hybrid Tiers + Cold/Warm Cache)
python scripts/run_benchmarks.py --iterations 3 --warmup 1 --plot

# 10. Render Pure-Python High-Resolution SVG Comparison Charts
python scripts/plot_benchmarks.py

# 11. Run Command-Line Clinical RAG Demonstration (Grounded Retrieval + Decoupled Evaluation)
python scripts/run_rag_demo.py --sample-suite

# 12. Run Ad-Hoc Clinical RAG Query with Department Filtering and Evaluation
python scripts/run_rag_demo.py "chest pain and dyspnea on exertion" --department Cardiology --evaluate

# 13. Run Clinical RAG in Retrieval-Only Mode (Generation Skipped, Passages Preserved)
python scripts/run_rag_demo.py "hypertension and elevated blood pressure" --provider retrieval-only --evaluate

# 14. Execute Controlled Vector-Index Empirical Benchmark (Flat vs HNSW vs IVFFlat)
python scripts/run_index_experiments.py --top-k 10 --iterations 3

# 15. Execute Ad-Hoc Hybrid Relational–Vector Query (with Department, Age, Document Type Filters)
python scripts/search_hybrid.py "shortness of breath and chest pain" --department Cardiology --document-type discharge_summary --min-age 40 --top-k 5 --explain

# 16. Run Multi-Selectivity Controlled Hybrid Benchmark (Low, Moderate, High, Zero-Match Tiers)
python scripts/run_hybrid_experiments.py --top-k 10 --iterations 3

# 17. Synchronize Synthetic Imaging Assets & Generate 512-d Multimodal Embeddings
python scripts/embed_images.py --smoke-test

# 18. Execute Image-to-Image Visual Similarity Search (Ranked by Cosine Distance)
python scripts/search_multimodal.py --image "data/synthetic/images/dx/syn-img-0020-01-01.png" --modality DX --top-k 5 --explain

# 19. Execute Cross-Modal Text-to-Image Semantic Search (Aligned CLIP Latent Space)
python scripts/search_multimodal.py --text "Chest CT showing bilateral infiltrates" --body-site Chest --top-k 5

# 20. Verify Database-Level Security, Least-Privilege Roles & Row-Level Security
python scripts/verify_security.py
```

For detailed architectural specifications:
- Data Ingestion & Sanitization: [`docs/data_ingestion.md`](file:///D:/HybridHealthcareVectorDB/docs/data_ingestion.md)
- Document Chunking & Vector Storage: [`docs/embedding_pipeline.md`](file:///D:/HybridHealthcareVectorDB/docs/embedding_pipeline.md)
- Ground-Truth Exact Retrieval Baseline: [`docs/exact_similarity_search.md`](file:///D:/HybridHealthcareVectorDB/docs/exact_similarity_search.md)
- Controlled Vector-Index Benchmark (Flat vs HNSW vs IVFFlat): [`docs/vector_index_experiments.md`](file:///D:/HybridHealthcareVectorDB/docs/vector_index_experiments.md)
- Hybrid Relational–Vector Query Execution & Selectivity: [`docs/hybrid_retrieval.md`](file:///D:/HybridHealthcareVectorDB/docs/hybrid_retrieval.md)
- Multimodal Storage, Vector Alignment & Cross-Modal Retrieval: [`docs/multimodal_retrieval.md`](file:///D:/HybridHealthcareVectorDB/docs/multimodal_retrieval.md)
- Database Security, Roles & Row-Level Security (RLS): [`docs/database_security.md`](file:///D:/HybridHealthcareVectorDB/docs/database_security.md)
- Security DDL & RLS Migration: [`sql/12_database_security_and_roles.sql`](file:///D:/HybridHealthcareVectorDB/sql/12_database_security_and_roles.sql)
- Reproducible Database Benchmarking Engine & Telemetry: [`docs/benchmarking_suite.md`](file:///D:/HybridHealthcareVectorDB/docs/benchmarking_suite.md)
- Command-Line Clinical RAG & Decoupled Evaluation Suite: [`docs/rag_demonstration.md`](file:///D:/HybridHealthcareVectorDB/docs/rag_demonstration.md)



