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
│   ├── 01_schema.sql         # Relational tables, constraints, and vector column definitions
│   ├── 02_indexes.sql        # B-tree, IVFFlat, and HNSW index creation scripts
│   ├── 03_hybrid_queries.sql # Attribute-first, vector-first, and joint hybrid queries
│   └── 04_security_rls.sql   # Role definitions and Row-Level Security policies
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
│   └── implementation_plan.md# Detailed phased development roadmap
└── scripts/                  # Standalone administrative and execution scripts
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
- [ ] **Phase 2:** PostgreSQL schema design, DDL definitions, and connection pooling.
- [ ] **Phase 3:** Synthetic healthcare dataset generator (encounters, notes, imaging metadata).
- [ ] **Phase 4:** Dense embedding generation and batch ingestion pipeline.
- [ ] **Phase 5:** Indexing implementation (Exact, IVFFlat, HNSW) and hybrid query engine.
- [ ] **Phase 6:** Row-Level Security (RLS) policies and role-based access testing.
- [ ] **Phase 7:** Rigorous experimental benchmark suite execution and result visualization.
- [ ] **Phase 8:** Command-line RAG demonstration and academic final evaluation report.
