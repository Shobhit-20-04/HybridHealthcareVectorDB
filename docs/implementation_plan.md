# Implementation Plan: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

## 1. Executive Summary & Research Motivation

In modern clinical informatics, healthcare data is fundamentally heterogeneous:
1. **Structured Data:** Patient demographics, vital signs, ICD-10 diagnostic codes, laboratory measurements, medication records, and encounter histories, which mandate strict ACID guarantees, relational integrity, and relational querying.
2. **Unstructured Data:** Free-text clinical discharge summaries, progress notes, pathology interpretations, and radiological imaging studies (e.g., Chest X-rays, MRI scans) requiring high-dimensional semantic search.

Traditional architectures isolate structured data in a relational DBMS (e.g., PostgreSQL, Oracle) while offloading embeddings to a dedicated external vector store (e.g., Pinecone, Milvus, Qdrant). This split-architecture introduces synchronization latency, dual-system operational overhead, cross-system consistency hazards, and inefficient two-phase query execution (requiring application-layer joins across network boundaries).

This academic DBMS project designs, implements, and benchmarks a **unified Hybrid Relational–Vector Database Architecture** centered on PostgreSQL with the `pgvector` extension. By unifying tabular clinical records and high-dimensional dense vector embeddings inside the same relational database engine, the architecture enables single-transaction atomicity, database-managed Row-Level Security (RLS), and single-pass hybrid query execution (combining relational filters with approximate nearest neighbor vector searches).

---

## 2. Architectural Blueprint

```
+---------------------------------------------------------------------------------+
|                       Client & Evaluation Layer                                 |
|   +--------------------------+  +-------------------------------------------+   |
|   | Synthetic Data Generator |  |  Automated Benchmark Engine (P50/P95/QPS) |   |
|   +--------------------------+  +-------------------------------------------+   |
|   +-------------------------------------------------------------------------+   |
|   |           Headless CLI Demonstration (Terminal-based Query & RAG)       |   |
|   +-------------------------------------------------------------------------+   |
+----------------------------------------|----------------------------------------+
                                         | Python 3.11 / psycopg3 / SQLAlchemy
+----------------------------------------v----------------------------------------+
|                          Application & Engine Layer                             |
|  +--------------------+  +----------------------+  +-------------------------+  |
|  | Embedding Pipeline |  | Hybrid Query Engine  |  | Security & Auth Context |  |
|  | - Text Transformers|  | - Pre-filter / Post- |  | - Role-based Context    |  |
|  | - Multimodal CLIP  |  |   filter optimization|  | - RLS Session Injection |  |
|  +--------------------+  +----------------------+  +-------------------------+  |
+----------------------------------------|----------------------------------------+
                                         | SQL + pgvector Operators (<->, <=>, <#>)
+----------------------------------------v----------------------------------------+
|                     PostgreSQL Unified Database Engine                          |
|  +-----------------------------------+  +------------------------------------+  |
|  | Relational Storage                |  | Vector Storage & Indexing          |  |
|  | - patients, encounters, diagnoses |  | - HNSW Index (m, ef_construction)  |  |
|  | - clinical_notes (JSONB metadata) |  | - IVFFlat Index (lists, probes)    |  |
|  | - imaging_studies (multimodal)    |  | - Exact Sequential Scan (Flat L2)  |  |
|  +-----------------------------------+  +------------------------------------+  |
|  +---------------------------------------------------------------------------+  |
|  | Engine Governance: Row-Level Security (RLS) & Multi-Tenant Access Policies |  |
|  +---------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------+
```

---

## 3. Module Specifications

### Module 1: Relational Schema & Document Storage
- **Relational Tables:**
  - `patients`: `patient_id` (PK, UUID), `mrn` (unique medical record number), `gender`, `date_of_birth`, `created_at`.
  - `encounters`: `encounter_id` (PK, UUID), `patient_id` (FK), `admission_date`, `discharge_date`, `encounter_type` (inpatient, outpatient, emergency), `department`.
  - `diagnoses`: `diagnosis_id` (PK, UUID), `encounter_id` (FK), `icd10_code`, `diagnosis_description`, `is_primary`.
  - `clinical_notes`: `note_id` (PK, UUID), `encounter_id` (FK), `patient_id` (FK), `note_type` (Discharge, Progress, Radiology, Pathology), `author_id`, `text_content`, `metadata` (`JSONB` for unstructured attributes, e.g., vital mentions, clinician annotations), `created_at`.
  - `clinical_embeddings`: `embedding_id` (PK, UUID), `note_id` (FK), `chunk_index`, `chunk_text`, `embedding` (`vector(384)` or `vector(768)`).
- **Relational Integrity:** Foreign key constraints, cascade rules, B-tree indexes on join keys (`patient_id`, `encounter_id`, `created_at`).

### Module 2: Vector Embeddings Pipeline
- **Text Embedding Engine:**
  - Pluggable architecture supporting HuggingFace sentence-transformers.
  - Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for high throughput.
  - Clinical alternative: Domain-adapted models (e.g., PubMedBERT-based sentence embeddings).
  - Deterministic chunking with configurable overlap for long clinical narratives.
- **Multimodal Embedding Engine:**
  - Multimodal projection for clinical studies linking imaging and text notes.
  - Image and report representations aligned in shared metric space (e.g., CLIP / BiomedCLIP `vector(512)`).

### Module 3: Vector Indexing & Search Modes
- **Exact Nearest Neighbor Search (Flat Scan):**
  - Sequential scan computing exact vector distances across the entire table.
  - Baseline gold standard for Recall@K calculations.
- **Inverted File Flat (IVFFlat) Index:**
  - Partitions vector space into Voronoi cells via k-means clustering.
  - Tunable parameters: `lists` (number of clusters), `probes` (number of clusters queried at runtime).
  - Trades indexing time and recall for query throughput.
- **Hierarchical Navigable Small World (HNSW) Index:**
  - Multi-layer proximity graph with logarithmic search scaling.
  - Tunable parameters: `m` (maximum edges per node), `ef_construction` (search depth during index building), `ef_search` (runtime search depth).
  - High recall with low query latency.
- **Distance Metrics Evaluated:**
  - Cosine Distance (`<=>`)
  - Euclidean / L2 Distance (`<->`)
  - Inner Product (`<#>`)

### Module 4: Hybrid SQL-Plus-Vector Query Processing
- **Pre-Filtering (Attribute-First):**
  - PostgreSQL applies relational B-Tree predicates first (e.g., `WHERE department = 'Cardiology' AND encounter_date >= '2025-01-01'`), followed by vector similarity ranking on the remaining candidate subset.
- **Post-Filtering (Vector-First):**
  - Executes vector index nearest neighbor retrieval to fetch top-$M$ ($M \gg K$), followed by relational predicate evaluation.
- **Iterative Index Scans & Joint Optimization:**
  - Analysis of PostgreSQL query planner execution plans (`EXPLAIN ANALYZE`), index scan choices, and query cost estimations across selectivity variations.

### Module 5: Multimodal Text/Image Embedding Storage
- **Schema:**
  - `imaging_studies`: `study_id` (PK, UUID), `patient_id` (FK), `modality` (CR, CT, MRI, DX), `body_site`, `study_timestamp`, `image_path`, `image_embedding` (`vector(512)`), `report_embedding` (`vector(512)`).
- **Cross-Modal Retrieval:**
  - Query by text to retrieve relevant diagnostic images.
  - Query by image embedding to retrieve historically similar diagnostic cases and verified findings.

### Module 6: Access Control & Security
- **HIPAA Compliance & Least Privilege:**
  - Multi-tenant role separation: `attending_physician`, `radiologist`, `clinical_researcher`, `auditor`.
- **PostgreSQL Row-Level Security (RLS):**
  - Policies restricting clinical record and vector retrieval based on session identity (`SET LOCAL app.current_user_role = 'radiologist'`).
  - Researchers restricted to de-identified records or cohorts with vector embeddings accessible only on sanitized corpora.
- **Audit Logging:**
  - Trigger-based or event-based audit trail tracking who queried which clinical records and semantic similarity searches.

### Module 7: Experimental Benchmarking Suite
- **Quantitative Metrics:**
  - **Latency:** Mean, Median ($P_{50}$), $P_{95}$, and $P_{99}$ latency across repeated query sets.
  - **Throughput:** Queries Per Second (QPS) under varying concurrency.
  - **Search Accuracy:** Recall@K relative to exact flat brute-force search.
  - **Index Build Metrics:** Wall-clock indexing duration, peak memory consumption, and disk footprint (bytes per vector and index size via `pg_relation_size`).
- **Controlled Experimental Dimensions:**
  - Scale: $10^3$, $10^4$, $10^5$ synthetic clinical records.
  - Selectivity: High selectivity (narrow filter, ~1% of rows matching) vs. Low selectivity (broad filter, ~50% of rows matching) vs. Unfiltered pure vector search.
  - Parameter sweeps: HNSW ($M \in \{8, 16, 32\}$, $ef\_construction \in \{32, 64, 128\}$, $ef\_search \in \{10, 20, 40, 80\}$) vs. IVFFlat ($lists \in \{50, 100, 200\}$, $probes \in \{1, 5, 10, 20\}$).

### Module 8: Command-Line RAG Demonstration (CLI Only)
- **Headless Retrieval-Augmented Generation Demonstration:**
  - Terminal-only command-line utility.
  - Accepts a natural language clinical inquiry (e.g., *"Patients with diabetic ketoacidosis refractory to standard insulin infusion"*).
  - Executes hybrid retrieval via PostgreSQL + pgvector, fetches ranked clinical context and relational metadata, and feeds formatted context into a localized CLI synthesis display.
  - Explicitly headless: No web server, no GUI, no dashboard.

---

## 4. Prerequisites & Environment Setup

### 4.1 System Prerequisites
- **Operating System:** Windows 10/11 x64
- **Python:** 3.11.x
- **PostgreSQL:** Version 16+ (PostgreSQL 18.x installed locally on system)
- **Extension:** `pgvector`

### 4.2 Installing `pgvector` on Windows PostgreSQL
Because PostgreSQL is installed at `C:\Program Files\PostgreSQL\18`, `pgvector` must be compiled or installed for PostgreSQL 18:
1. **Option A (Precompiled Extension Binaries):**
   - Download the compatible `pgvector` release archive for Windows/PostgreSQL 18.
   - Copy `vector.dll` into `C:\Program Files\PostgreSQL\18\lib`.
   - Copy `vector.control` and `vector--*.sql` into `C:\Program Files\PostgreSQL\18\share\extension`.
2. **Option B (Build from Source via MSVC):**
   - Ensure Visual Studio C++ Build Tools (MSVC `nmake` and `cl`) are in PATH.
   - Set environment variable: `$env:PGROOT = "C:\Program Files\PostgreSQL\18"`
   - Clone repository: `git clone https://github.com/pgvector/pgvector.git`
   - Build: `nmake /F Makefile.msvc`
   - Install: `nmake /F Makefile.msvc install`
3. **Database Activation:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

---

## 5. Phased Roadmap

| Phase | Title | Scope | Status |
|---|---|---|---|
| **Phase 1** | Project Initialization & Environment Definition | Directory hierarchy, requirements, `.env.example`, `.gitignore`, architectural plan | Completed |
| **Phase 2** | Database Configuration & Relational Schema | PostgreSQL DDL (`01_extensions.sql` - `04_seed_embedding_models.sql`), schema docs, validator | Completed |
| **Phase 3** | Synthetic Healthcare Dataset Pipeline | Generator for clinical notes, encounters, diagnoses, and multimodal metadata | Completed |
| **Phase 4** | Embedding Generation & Ingestion Engine | Batch text embedding, CLIP image vector pipeline, pgvector storage | Completed |
| **Phase 5** | Indexing Strategies & Hybrid Query Engine | Exact Flat, IVFFlat, HNSW implementation; Pre-filter & Post-filter SQL queries | Completed |
| **Phase 6** | Security Architecture & Row-Level Security | PostgreSQL RLS policies, role simulation scripts, de-identification views | Completed |
| **Phase 7** | Empirical Benchmarking Suite | Rigorous measurement of Recall@K, latency, throughput, index size, output tables/graphs | Completed |
| **Phase 8** | Terminal CLI Demonstration & Documentation | Interactive headless CLI, reproducible run instructions, final academic summary | Completed |

> [!IMPORTANT]
> In accordance with the academic guidelines, all benchmark data will be empirically measured through script runs rather than invented. No web UI or dashboard will be constructed.
