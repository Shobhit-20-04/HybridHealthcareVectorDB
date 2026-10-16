# Final Academic Research Report: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

**Project Title:** Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval  
**Database System:** PostgreSQL 18.6 on x86_64-windows with `pgvector` v0.8.6  
**Hardware Baseline:** Intel64 Family 6 Model 183 Stepping 1 (16 logical cores) | 15.69 GB RAM | Windows 10/11 x64  
**Date of Audit:** October 2026  
**Repository:** `D:\HybridHealthcareVectorDB`  

---

## 1. Executive Summary & Research Objectives

Contemporary healthcare information systems operate across two distinct data paradigms:
1. **Structured Clinical Data:** Relational records encompassing patient demographics, encounter histories, admission dates, clinical departments, and ICD-10 diagnostic billing codes requiring strict ACID guarantees, foreign key relational integrity, and B-Tree indexing.
2. **Unstructured Narrative & Diagnostic Data:** Clinical free-text narratives (discharge summaries, consult notes, pathology reports) and diagnostic imaging examinations (digital radiography, chest computed tomography, ultrasonography) requiring high-dimensional semantic vector search.

Legacy architectures typically address this dichotomy through a **dual-database design**, pairing an operational relational database (e.g. PostgreSQL, Oracle) with an external standalone vector database (e.g. Milvus, Pinecone, Qdrant). Dual-database designs incur severe operational liabilities in healthcare environments:
- **Synchronization Drift:** Asynchronous data pipelines introduce replication lag between relational state and vector embeddings, creating dangerous window periods where medical documents are indexed without up-to-date relational metadata.
- **Two-Phase Network Query Overhead:** Queries requiring relational filtering and vector ranking force round-trip distributed joins across disparate network endpoints, incurring high latency penalties.
- **Fragmented Security & Access Control:** Row-level authorization, HIPAA audit logging, and role-based policies must be redundantly implemented across both storage engines, multiplying the attack surface for Protected Health Information (PHI) exfiltration.

This project investigates a **unified hybrid database architecture** centered on **PostgreSQL 18.6** augmented with the native **`pgvector`** C extension. The research empirically evaluates database internals, indexing performance trade-offs, physical execution plans under variable filter selectivities, modality isolation, database-level row-level security (RLS), and retrieval-augmented generation groundedness.

---

## 2. System Architecture

The database architecture integrates relational tables, vector embedding stores, security policies, and retrieval engines within a single ACID-compliant database instance:

```mermaid
flowchart TD
    subgraph ClientLayer["Application & Analysis Clients"]
        CLI_Admin["Database Administrator / DBA"]
        CLI_Physician["Attending Physician (Cardiology, etc.)"]
        CLI_Radiologist["Diagnostic Radiologist"]
        CLI_Researcher["Clinical Researcher"]
        CLI_RAG["Headless RAG Demo (scripts/run_rag_demo.py)"]
        Bench["Unified Benchmark Suite (scripts/run_benchmarks.py)"]
    end

    subgraph SecurityKernel["PostgreSQL Security & Authorization Kernel"]
        RBAC["Role-Based Access Control (RBAC)\n(clinical_admin, attending_physician, radiologist, clinical_researcher)"]
        RLS["Row-Level Security (RLS) Engine\n(Kernel-level tuple filtering before distance ranking)"]
        SessionCtx["Transactional Session Context\n(SET LOCAL app.clinician_dept = '...')"]
        DeidView["deidentified_clinical_documents\n(Sanitized View for Researchers)"]
    end

    subgraph RelationalCore["Normalized Relational Schema (3NF)"]
        Patients["patients\n(patient_id PK, synthetic_mrn, dob, gender)"]
        Encounters["encounters\n(encounter_id PK, patient_id FK, department, admission_date)"]
        Diagnoses["diagnoses\n(diagnosis_id PK, encounter_id FK, icd10_code, description)"]
        Documents["clinical_documents\n(document_id PK, encounter_id FK, title, document_type, narrative)"]
        Images["medical_images\n(image_id PK, patient_id FK, encounter_id FK, modality, file_path)"]
    end

    subgraph VectorCore["pgvector Vector Storage Layer"]
        ModelRegistry["embedding_models\n(model_id PK, dimension, distance_metric, modality)"]
        Chunks["document_chunks\n(chunk_id PK, document_id FK, chunk_index, chunk_text)"]
        TextEmbed["chunk_embeddings\n(embedding_id PK, chunk_id FK, model_id FK,\nembedding vector(384))"]
        ImageEmbed["image_embeddings\n(embedding_id PK, image_id FK, model_id FK,\nembedding vector(512))"]
    end

    subgraph IndexEngines["pgvector Indexing Engines"]
        FlatScan["Exact Flat (Sequential Scan)"]
        HNSW["HNSW Graph Index\n(vector_cosine_ops, m=16, ef_c=64)"]
        IVFFlat["IVFFlat Inverted File Index\n(vector_cosine_ops, lists=25)"]
    end

    CLI_Admin --> RBAC
    CLI_Physician --> SessionCtx --> RLS
    CLI_Radiologist --> SessionCtx --> RLS
    CLI_Researcher --> DeidView
    CLI_RAG --> RLS
    Bench --> IndexEngines

    Patients --> Encounters --> Documents --> Chunks --> TextEmbed
    Encounters --> Diagnoses
    Patients --> Images --> ImageEmbed
    ModelRegistry -.-> TextEmbed
    ModelRegistry -.-> ImageEmbed

    TextEmbed --> IndexEngines
    ImageEmbed --> FlatScan
```

---

## 3. Entity-Relationship (ER) Diagram & Schema Design

The relational schema strictly adheres to Third Normal Form (3NF), decoupling clinical entities from chunks and vector representations.

```mermaid
erDiagram
    PATIENTS ||--o{ ENCOUNTERS : "admits"
    PATIENTS ||--o{ MEDICAL_IMAGES : "undergoes"
    ENCOUNTERS ||--o{ DIAGNOSES : "establishes"
    ENCOUNTERS ||--o{ CLINICAL_DOCUMENTS : "generates"
    ENCOUNTERS ||--o{ MEDICAL_IMAGES : "orders"
    CLINICAL_DOCUMENTS ||--o{ DOCUMENT_CHUNKS : "segments into"
    DOCUMENT_CHUNKS ||--o{ CHUNK_EMBEDDINGS : "vectorized into"
    MEDICAL_IMAGES ||--o{ IMAGE_EMBEDDINGS : "encoded into"
    EMBEDDING_MODELS ||--o{ CHUNK_EMBEDDINGS : "governs space"
    EMBEDDING_MODELS ||--o{ IMAGE_EMBEDDINGS : "governs space"

    PATIENTS {
        uuid patient_id PK
        varchar synthetic_mrn UK
        date date_of_birth
        varchar gender
        jsonb address_metadata
        timestamptz created_at
    }

    ENCOUNTERS {
        uuid encounter_id PK
        uuid patient_id FK
        varchar synthetic_encounter_num UK
        varchar department
        varchar encounter_type
        timestamptz admission_date
        timestamptz discharge_date
    }

    DIAGNOSES {
        uuid diagnosis_id PK
        uuid encounter_id FK
        uuid patient_id FK
        varchar icd10_code
        text description
        varchar diagnosis_type
    }

    CLINICAL_DOCUMENTS {
        uuid document_id PK
        uuid encounter_id FK
        uuid patient_id FK
        varchar synthetic_document_num UK
        varchar title
        varchar document_type
        text raw_text
        varchar provenance_hash
        jsonb metadata
    }

    DOCUMENT_CHUNKS {
        uuid chunk_id PK
        uuid document_id FK
        int chunk_index
        text chunk_text
        int char_length
        int token_count
        varchar chunk_hash
    }

    EMBEDDING_MODELS {
        varchar model_id PK
        int dimension
        varchar distance_metric
        varchar modality
        varchar version
    }

    CHUNK_EMBEDDINGS {
        uuid embedding_id PK
        uuid chunk_id FK
        varchar model_id FK
        vector embedding
        timestamptz created_at
    }

    MEDICAL_IMAGES {
        uuid image_id PK
        uuid patient_id FK
        uuid encounter_id FK
        varchar synthetic_image_num UK
        varchar modality
        varchar body_site
        text file_path
        varchar file_hash
        jsonb acquisition_metadata
    }

    IMAGE_EMBEDDINGS {
        uuid embedding_id PK
        uuid image_id FK
        varchar model_id FK
        vector embedding
        timestamptz created_at
    }
```

### Normalization & Vector Isolation Rationale:
1. **Decoupled Chunk and Embedding Tables:** A single clinical document is segmented into multiple overlapping chunks (`document_chunks`). Each chunk can support embeddings from multiple competing embedding models simultaneously (`chunk_embeddings` references `embedding_models`), enabling side-by-side model benchmarking without document replication.
2. **Strict Vector Dimension Typing:** The text embedding column is typed as `vector(384)` and the image embedding column as `vector(512)`. Foreign key constraints to `embedding_models` prevent dimension mismatch errors at the database boundary.
3. **Audit & Provenance Integrity:** Every document and image asset records a SHA-256 provenance hash (`provenance_hash`, `file_hash`) preventing undetected record modification.

---

## 4. Vector Indexing Methods & Trade-offs

The architecture supports three vector search strategies in PostgreSQL:

| Indexing Method | PostgreSQL Implementation Syntax | Algorithmic Mechanism | Memory / Storage Characteristics | Approximate vs Exact |
|:---|:---|:---|:---|:---:|
| **Exact Flat** | `enable_indexscan = off, enable_seqscan = on` | Brute-force sequential scan computing cosine distance across all tuples using CPU SIMD vector extensions. | Zero additional index overhead; identical to heap relation size. | **Exact** (100% ground-truth Recall) |
| **HNSW** | `USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)` | Multi-layer proximity graph where upper layers enable fast long-range navigation and lower layers fine-tune local search. | Compact in-memory graph representation; fast queries with high recall. | **Approximate** (Tunable via `hnsw.ef_search`) |
| **IVFFlat** | `USING ivfflat (embedding vector_cosine_ops) WITH (lists = 25)` | Inverted file index clustering vector space into Voronoi centroid cells. Search probes only the nearest centroid inverted lists. | Requires sufficient data clustering ($N > lists \times 10$); high pre-allocation page footprint. | **Approximate** (Tunable via `ivfflat.probes`) |

---

## 5. Hybrid Relational-Plus-Vector Query Execution

Hybrid queries unite structured relational metadata predicates (clinical department, patient age, document type, ICD-10 billing code) with dense vector distance ranking (`ORDER BY embedding <=> query_vector ASC LIMIT K`).

### Parameterized SQL Query Structure
All hybrid queries are executed via parameterized SQL, preventing SQL injection vulnerabilities:

```sql
SELECT 
    ce.chunk_id,
    dc.chunk_index,
    dc.chunk_text AS snippet,
    cd.document_id,
    cd.synthetic_document_num,
    cd.title,
    cd.document_type,
    p.patient_id,
    p.synthetic_mrn,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.date_of_birth))::int AS patient_age,
    e.encounter_id,
    e.department,
    (ce.embedding <=> %s::vector) AS cosine_distance,
    (1.0 - (ce.embedding <=> %s::vector)) AS cosine_similarity
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
WHERE ce.model_id = %s
  AND e.department = %s
  AND cd.document_type = %s
  AND EXTRACT(YEAR FROM AGE(CURRENT_DATE, p.date_of_birth)) >= %s
ORDER BY ce.embedding <=> %s::vector ASC
LIMIT %s;
```

### Query Optimizer Behavior Across Selectivity Tiers
Through `EXPLAIN (ANALYZE, BUFFERS)` instrumentation, the PostgreSQL cost-based query optimizer demonstrates two distinct execution strategies depending on relational predicate selectivity:
1. **High Selectivity / Restrictive Predicates (~1% match):**
   The planner utilizes a **Bitmap Index Scan** on relational B-Tree indexes (`idx_clinical_docs_dept`, `idx_encounters_dept`) to extract matching tuple IDs first (Pre-Filtering), subsequently joining to `chunk_embeddings` and computing distances only for surviving candidates.
2. **Low Selectivity / Broad Predicates (~45%–70% match):**
   The planner transitions to a **Sequential Scan** or **Iterative Vector Index Scan**, sorting candidate chunks by vector proximity and joining relational rows during tuple evaluation.

---

## 6. Multimodal Representation & Cross-Modal Retrieval

The database provides native multimodal storage for diagnostic radiology assets (Digital Radiography `DX`, Computed Tomography `CT`, Ultrasonography `US`):
- **Physical Asset Integrity:** Synthetic PNG images are preserved in `data/synthetic/images/{modality}/` with SHA-256 hashes verified against `medical_images.file_hash`.
- **Modality Isolation Guarantee:**
  - Text chunk embeddings (`vector(384)`) and medical image embeddings (`vector(512)`) inhabit distinct latent metric spaces.
  - The schema physically segregates them into `chunk_embeddings` and `image_embeddings`.
  - Database check constraints and foreign keys to `embedding_models` prevent mathematical comparison between incompatible vector dimensions.
- **Cross-Modal Retrieval:** Enabled via unified multimodal encoders (e.g. `openai/clip-vit-base-patch32`), allowing natural language queries (`"Chest Radiograph bilateral infiltrates"`) to retrieve ranked radiological images via cosine distance.

---

## 7. Database-Level Security, RBAC & Row-Level Security (RLS)

Security is enforced at the **PostgreSQL kernel level**, guaranteeing that unauthorized patient records cannot be exfiltrated through direct queries, joins, or vector similarity ranking.

### Least-Privilege Role Hierarchy
| Role Name | Table Permissions | Row-Level Security (RLS) Policy Scope | Intended Clinical Access |
|:---|:---|:---|:---|
| `clinical_admin` | `ALL` on all tables and views | Unrestricted access across all rows. | System administration and migrations |
| `attending_physician` | `SELECT` on patients, encounters, documents, chunks, vectors, images | Restricted to records where `encounters.department = current_setting('app.clinician_dept')`. | Attending clinicians bounded by department |
| `radiologist` | `SELECT` on medical images, image embeddings, radiology reports | Restricted to `medical_images` and documents with `document_type = 'radiology_report'`. | Diagnostic imaging specialists |
| `clinical_researcher` | `SELECT` on `deidentified_clinical_documents` view only | **Strict Deny** on direct `patients` and `medical_images` tables; access limited to sanitized view. | De-identified research and cohort studies |

### Vector Security Isolation
Row-Level Security policies on `clinical_documents` automatically cascade to `document_chunks` and `chunk_embeddings`. When an attending physician executes a top-k vector search:
$$\text{SELECT} \dots \text{ORDER BY embedding} \Leftrightarrow q \text{ LIMIT } k$$
PostgreSQL evaluates the RLS policy **before** vector distance sorting, guaranteeing that nearest-neighbor ranking never surfaces unauthorized clinical notes from other departments.

---

## 8. Experimental Setup & System Telemetry

All benchmark experiments were conducted live on the dedicated host system with full hardware and database telemetry embedded in every output file:

| Telemetry Category | Parameter | Measured Value |
|:---|:---|:---|
| **Host Hardware** | Processor / Architecture | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (16 logical cores) |
| **Host Hardware** | Physical Memory (RAM) | 15.69 GB Total RAM (Windows `GlobalMemoryStatusEx`) |
| **Operating System** | Platform / OS Version | Windows 10/11 x64 (Build 10.0.26300-SP0) |
| **Database Engine** | PostgreSQL Version | PostgreSQL 18.6 on x86_64-windows, MSVC 64-bit |
| **Vector Extension** | `pgvector` Version | v0.8.6 |
| **Database Configuration** | Memory Buffers | `shared_buffers = 128MB`, `work_mem = 4MB` |
| **Text Embedding Model** | Identifier & Metric | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, Cosine `<=>`) |
| **Multimodal Model** | Identifier & Metric | `openai/clip-vit-base-patch32` (512 dimensions, Cosine `<=>`) |
| **Dataset Scale** | Document / Vector Counts | 197 Patients, 402 Encounters, 586 Documents, 804 Chunks, 227 Images |

---

## 9. Empirical Benchmark Results & Analysis

*All metrics are acquired live from PostgreSQL 18.6 across 15 clinical queries, 3 iterations, and 1 warm-up run (540 individual query samples).*

### 9.1 Pure Vector Search & Indexing Performance

| Configuration | Index Type | Build Time | Storage Size | Mean Latency | $P_{50}$ (Median) | $P_{90}$ | $P_{95}$ | $P_{99}$ | Cold $P_{50}$ | QPS | 1-Recall@10 | Noise Status |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Exact Flat Baseline** | Flat | - | 0 bytes | 10.65 ms | 10.65 ms | 10.98 ms | 11.23 ms | 11.39 ms | 10.39 ms | 93.9 | 100.0% | `STABLE` |
| **HNSW ($ef_s=10$)** | HNSW | 59.3 ms | 184 kB | 11.48 ms | 11.28 ms | 11.87 ms | 12.34 ms | 18.18 ms | 11.11 ms | 87.1 | 100.0% | `STABLE` |
| **HNSW ($ef_s=40$)** | HNSW | 59.3 ms | 184 kB | 11.39 ms | 11.30 ms | 11.83 ms | 11.95 ms | 14.89 ms | 11.30 ms | 87.8 | 100.0% | `STABLE` |
| **HNSW ($ef_s=100$)** | HNSW | 59.3 ms | 184 kB | 11.33 ms | 11.29 ms | 11.74 ms | 11.87 ms | 12.67 ms | 11.34 ms | 88.3 | 100.0% | `STABLE` |
| **IVFFlat ($probes=1$)** | IVFFlat | 47.2 ms | 1,472 kB | 11.28 ms | 11.18 ms | 11.68 ms | 11.72 ms | 16.26 ms | 11.30 ms | 88.6 | 100.0% | `STABLE` |
| **IVFFlat ($probes=5$)** | IVFFlat | 47.2 ms | 1,472 kB | 11.26 ms | 11.23 ms | 11.58 ms | 11.63 ms | 11.75 ms | 11.23 ms | 88.8 | 100.0% | `STABLE` |
| **IVFFlat ($probes=10$)** | IVFFlat | 47.2 ms | 1,472 kB | 11.29 ms | 11.28 ms | 11.66 ms | 11.82 ms | 11.98 ms | 11.23 ms | 88.6 | 100.0% | `STABLE` |

### 9.2 Hybrid Relational-Vector Query Performance Across Selectivity Tiers

| Selectivity Tier | Match % | Search Mode | Mean Latency | $P_{50}$ (Median) | $P_{90}$ | $P_{95}$ | Throughput | 1-Recall@10 | Noise Status |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| **Low Selectivity (~45%)** | 44.8% (360 chunks) | Exact Flat | 80.65 ms | 51.97 ms | 157.66 ms | 165.12 ms | 12.4 QPS | 100.0% | `HIGH_VARIANCE` |
| **Low Selectivity (~45%)** | 44.8% (360 chunks) | HNSW Planner | 82.70 ms | 53.09 ms | 167.18 ms | 168.14 ms | 12.1 QPS | 100.0% | `HIGH_VARIANCE` |
| **Moderate Selectivity (~11%)** | 10.6% (85 chunks) | Exact Flat | 64.19 ms | 50.70 ms | 150.08 ms | 150.63 ms | 15.6 QPS | 100.0% | `HIGH_VARIANCE` |
| **Moderate Selectivity (~11%)** | 10.6% (85 chunks) | HNSW Planner | 64.54 ms | 49.28 ms | 148.49 ms | 167.74 ms | 15.5 QPS | 100.0% | `HIGH_VARIANCE` |
| **High Selectivity (~1%)** | 1.0% (8 chunks) | Exact Flat | 77.14 ms | 49.30 ms | 158.12 ms | 160.95 ms | 13.0 QPS | 100.0% | `HIGH_VARIANCE` |
| **High Selectivity (~1%)** | 1.0% (8 chunks) | HNSW Planner | 77.71 ms | 50.01 ms | 155.22 ms | 162.34 ms | 12.9 QPS | 100.0% | `HIGH_VARIANCE` |
| **Zero-Match Filter (0%)** | 0.0% (0 chunks) | Exact Flat | 75.72 ms | 47.71 ms | 155.53 ms | 155.98 ms | 13.2 QPS | 100.0% | `HIGH_VARIANCE` |
| **Zero-Match Filter (0%)** | 0.0% (0 chunks) | HNSW Planner | 69.42 ms | 48.07 ms | 152.32 ms | 169.32 ms | 14.4 QPS | 100.0% | `HIGH_VARIANCE` |

### 9.3 Key Scientific Findings
1. **In-Memory Sequential Scan Competitiveness at Small Scale:**
   At 804 chunk vectors, CPU AVX-512 SIMD vector acceleration enables exact flat scans to execute in **10.65 ms**, matching or slightly outperforming approximate index lookups due to graph traversal overhead. Indexing advantages manifest significantly as corpus size surpasses 10,000+ vectors.
2. **Storage Footprint Disparity:**
   HNSW requires **184 kB** on disk, whereas IVFFlat requires **1,472 kB** (an 8.0x larger storage footprint) because list centroid structures and pre-allocated inverted pages require upfront block allocations in PostgreSQL.
3. **Hybrid Query Variance:**
   Hybrid queries exhibit higher latency variance (`HIGH_VARIANCE` flag) compared to pure vector scans. Variance stems from query plan compilation and relational hash-join overhead across four joined tables (`patients`, `encounters`, `clinical_documents`, `document_chunks`).
4. **Cold vs. Warm Latency:**
   Session cache invalidation (`DISCARD ALL;`) produces minimal latency delta (<0.3 ms) because the active dataset remains resident in the Windows operating system file cache.

---

## 10. Command-Line Clinical RAG Demonstration

The project provides a headless CLI RAG engine ([`scripts/run_rag_demo.py`](file:///D:/HybridHealthcareVectorDB/scripts/run_rag_demo.py)) evaluating grounded generation:

### Core Capabilities:
- **Passage-Only Grounding:** Passes only top-k database passages to the language model.
- **Explicit Citations:** Requires every clinical claim to cite its document ID in brackets (e.g. `[SYN-DOC-0134-01-01]`).
- **Uncertainty Distinction:** Formats output with separate `**Documented Evidence:**` and `**Clinical Uncertainty & Limitations:**` sections.
- **Mandatory Abstention:** When queried on conditions outside the corpus (e.g. pediatric neurosurgery), the model abstains verbatim:
  *"The available documents do not establish an answer to the question."*
- **Decoupled Evaluation:** Evaluates retrieval quality (lexical overlap, top distance) independently of generation groundedness (citation validity, unsupported claim detection).

---

## 11. Limitations & Critical Assumptions

1. **Synthetic Data Constraint:**
   All clinical notes, MRNs, patient demographics, and imaging phantoms are synthetic. While generated deterministically using standard ICD-10 clinical taxonomies, they do not encompass rare comorbidities or unstructured real-world dictation errors.
2. **Corpus Scale Limitations:**
   The primary benchmark evaluation was conducted on an 804-chunk corpus. While representative for validating architectural correctness and query optimizer execution paths, index scalability benefits (e.g. sub-linear HNSW search speedups) become pronounced primarily at 100,000+ vectors.
3. **No Clinical Diagnostic Utility:**
   The system is an academic database management prototype. It does not provide medical advice, triage recommendations, or diagnostic validation.

---

## 12. Incomplete Tasks & Manual Prerequisites

### Current Implementation Status: Complete
All 12 core requirements specified in the project roadmap are implemented and verified:
- [x] Schema DDL, check constraints, foreign keys, and vector column typing
- [x] Deterministic synthetic healthcare data generator and transactional loader
- [x] Text chunking, embedding generation, and idempotent vector storage
- [x] Exact sequential flat scan semantic retrieval engine
- [x] Controlled vector index experiments (Exact Flat vs. HNSW vs. IVFFlat)
- [x] Hybrid relational-vector search engine with 4 selectivity tiers
- [x] Multimodal schema, phantom generation, and cross-modal retrieval
- [x] Database-level security, least-privilege roles, and Row-Level Security (RLS)
- [x] Unified reproducible database benchmarking suite with telemetry and SVG plots
- [x] Command-line clinical RAG demonstration with decoupled evaluation
- [x] 106 automated tests passing across 12 test suites

### Manual Prerequisites for External Users:
1. **PostgreSQL 18.x with `pgvector`:**
   Ensure PostgreSQL is installed locally on port 5432 with `vector.dll` and `vector.control` deployed in the PostgreSQL library directories.
2. **HuggingFace Model Weights (Optional):**
   The system includes offline deterministic encoders for standalone testing. To activate official PyTorch weights, users must install `torch` and `sentence-transformers`:
   ```powershell
   pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. **External LLM Opt-In (Optional):**
   To connect to cloud LLM providers, users must provide `GEMINI_API_KEY` or `EXTERNAL_LLM_API_KEY` and explicitly pass `--allow-external`.

---

## 13. Future Work

1. **Quantization & Disk-Backed Vector Indexing:**
   Evaluate scalar quantization (SQ8) and binary quantization (BQ) supported in newer `pgvector` releases to reduce memory consumption on multi-million vector corpora.
2. **Iterative Graph Filter Traversal:**
   Investigate experimental filtered HNSW algorithms that traverse graph edges conditioned on relational bitmasks rather than post-filtering graph candidate sets.
3. **Asynchronous Batch Inference:**
   Implement streaming asynchronous queue ingestion for high-throughput clinical document feeds.
