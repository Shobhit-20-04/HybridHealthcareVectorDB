# Comprehensive Demonstration & Reviewer Checklist

**Project:** Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval  
**Working Directory:** `D:\HybridHealthcareVectorDB`  
**Platform:** Windows 10/11 x64  
**Database System:** PostgreSQL 18.6 with `pgvector` v0.8.6  
**Python Runtime:** Python 3.11.x  

---

## 1. System Prerequisites Checklist

Before executing the demonstration, verify that the following prerequisites are installed and operational:

- [ ] **Python 3.11.x Installed:** Accessible in PowerShell via `python --version`.
- [ ] **PostgreSQL 18.x Running:** Service active on `localhost:5432`.
- [ ] **`pgvector` Extension Installed:** `vector.dll` in PostgreSQL `lib/` directory and `vector.control` in `share/extension/`.
- [ ] **Environment Configuration (`.env`):** Present in `D:\HybridHealthcareVectorDB\.env` with database credentials:
  ```ini
  DB_HOST=localhost
  DB_PORT=5432
  DB_NAME=healthcare_vectordb
  DB_USER=postgres
  DB_PASSWORD=your_password_here
  ```

---

## 2. Step-by-Step Demonstration Execution

Follow each command in sequence from a Windows PowerShell terminal at `D:\HybridHealthcareVectorDB`:

### Step 1: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*Expected Result:* Terminal prompt displays `(.venv)`.

---

### Step 2: Initialize Database & Run Schema Migrations
```powershell
python scripts/init_database.py
```
*Expected Result:*
- Connects to PostgreSQL at `localhost:5432/healthcare_vectordb`.
- Executes 13 SQL migration files in order (Extensions, Schema, Indexes, Model Registry, Security Roles, RLS policies).
- Displays `[V] Database migrations completed successfully.`

---

### Step 3: Validate SQL Schema & Invariant Integrity
```powershell
python scripts/validate_sql.py
```
*Expected Result:*
- Parses 10 tables, 28 indexes, 13 foreign keys, and 3 vector columns (`chunk_embeddings.embedding -> vector(384)`, `image_embeddings.embedding -> vector(512)`).
- Outputs `[V] ALL SQL SCHEMA CHECKS PASSED SUCCESSFULLY.`

---

### Step 4: Generate Deterministic Synthetic Healthcare Data
```powershell
python scripts/generate_data.py --preset small --seed 42
```
*Expected Result:*
- Generates reproducible cohort: 200 patients, 406 encounters, 693 diagnoses, 591 clinical documents, 810 chunks, 230 imaging records.
- Saves sanitized JSON datasets into `data/synthetic/`.

---

### Step 5: Transactionally Load Data into PostgreSQL
```powershell
python scripts/load_data.py --truncate
```
*Expected Result:*
- Transactionally inserts patients, encounters, diagnoses, clinical documents, document chunks, and medical images.
- Validates foreign keys, provenance hashes, and uniqueness constraints.
- Commits transaction and outputs `[V] Ingestion completed successfully.`

---

### Step 6: Verify Database Ingestion & Referential Integrity
```powershell
python scripts/validate_ingestion.py
```
*Expected Result:*
- Checks for orphan records across all 8 foreign key relationships (0 orphans found).
- Checks chunk sequences, dates, and provenance hashes.
- Outputs `[V] ALL INGESTION INTEGRITY AND REFERENTIAL CHECKS PASSED.`

---

### Step 7: Generate Dense Text Embeddings & Store in pgvector
```powershell
python scripts/embed_documents.py --smoke-test
```
*Expected Result:*
- Reads 804 document chunks from PostgreSQL.
- Generates 384-dimensional vector embeddings and writes them to `chunk_embeddings`.
- Confirms vector dimension matches column type; idempotent reruns create zero duplicate embeddings.

---

### Step 8: Generate Diagnostic Image Phantoms & Embeddings
```powershell
python scripts/embed_images.py --smoke-test
```
*Expected Result:*
- Generates synthetic DICOM-like PNG image phantoms for DX, CT, and US examinations.
- Generates 512-dimensional vector embeddings and writes them to `image_embeddings`.
- Verifies physical asset paths on disk.

---

### Step 9: Audit Live Database Table Counts
```powershell
python scripts/count_records.py
```
*Expected Result:*
- Displays table audit: 197 patients, 402 encounters, 685 diagnoses, 586 documents, 804 chunks, 804 text embeddings, 227 images, 227 image embeddings (3,936 total rows).

---

### Step 10: Run Exact Semantic Search (Ground-Truth Baseline)
```powershell
python scripts/search_exact.py "progressive dyspnea on exertion" --top-k 3
```
*Expected Result:*
- Executes brute-force sequential flat scan (`enable_indexscan=off`).
- Returns top-3 ranked chunks with cosine distances (lower is closer), snippets, titles, and timing breakdown (~10–50 ms).

---

### Step 11: Execute Controlled Vector Index Experiments (Flat vs HNSW vs IVFFlat)
```powershell
python scripts/run_index_experiments.py --top-k 10 --iterations 3
```
*Expected Result:*
- Tests Exact Flat, HNSW ($ef_s=10, 40, 100$), and IVFFlat ($probes=1, 5, 10$).
- Measures index build time, physical size (`pg_relation_size`), query latency ($P_{50}, P_{95}$), and 1-Recall@K against exact baseline.
- Saves results to `results/vector_index_benchmark.json`.

---

### Step 12: Run Hybrid Relational–Vector Search
```powershell
python scripts/search_hybrid.py "progressive dyspnea on exertion" --department Cardiology --top-k 3
```
*Expected Result:*
- Combines semantic vector similarity with relational SQL filter `department = 'Cardiology'`.
- Returns only Cardiology documents with complete joined patient, encounter, and ICD-10 diagnostic metadata.

---

### Step 13: Execute Multi-Selectivity Hybrid Benchmark
```powershell
python scripts/run_hybrid_experiments.py --top-k 10 --iterations 3
```
*Expected Result:*
- Evaluates 4 filter selectivity tiers: Low (~45%), Moderate (~11%), High (~1%), and Zero-Match (0%).
- Demonstrates optimizer execution plan transitions (Pre-Filtering Bitmap Scans vs Iterative Vector Scans).
- Saves results to `results/hybrid_query_benchmark.json`.

---

### Step 14: Run Multimodal Cross-Modal Search
```powershell
python scripts/search_multimodal.py --text "Chest Radiograph bilateral infiltrates" --top-k 3
```
*Expected Result:*
- Natural language text query retrieves ranked diagnostic images via 512-d CLIP metric space.
- Displays Modality (DX), Body Site (Chest), Patient MRN, Department, Distance, and physical file path verification.

---

### Step 15: Verify Database-Level Security & Row-Level Security (RLS)
```powershell
python scripts/verify_security.py
```
*Expected Result:*
- Audits 4 PostgreSQL roles (`clinical_admin`, `attending_physician`, `radiologist`, `clinical_researcher`) and active RLS policies.
- Executes 9 positive and negative authorization tests (Cardiology physician cannot view Neurology notes; Researcher cannot query `patients` table directly).
- Outputs `[V] ALL 9 AUTHORIZATION & SECURITY TESTS PASSED SUCCESSFULLY!`

---

### Step 16: Run Unified Benchmark Suite & Render SVG Charts
```powershell
python scripts/run_benchmarks.py --iterations 3 --warmup 1 --plot
```
*Expected Result:*
- Runs full 540-query benchmark across Flat, HNSW, IVFFlat, and all 4 hybrid selectivity tiers.
- Embeds hardware and database telemetry into all output files.
- Generates `results/benchmark_report.json`, `results/benchmark_summary.csv`, `results/benchmark_summary.md`, and 3 SVG vector charts.

---

### Step 17: Run Command-Line Clinical RAG Demonstration
```powershell
python scripts/run_rag_demo.py --sample-suite
```
*Expected Result:*
- Executes 3 clinical demonstration cases with decoupled evaluation:
  1. Supported Cardiology condition: Grounded response with `[SYN-DOC-xxxx]` citations and uncertainty distinction.
  2. Pulmonology filter: High-selectivity hybrid retrieval.
  3. Pediatric neurosurgery (Negative query): Abstains verbatim with *"The available documents do not establish an answer to the question."*
- Generates decoupled Retrieval Quality and Generation Groundedness evaluation reports.

---

### Step 18: Run Complete Automated Test Suite
```powershell
pytest tests/ -v
```
*Expected Result:*
- Executes 106 automated unit, integration, security, multimodal, benchmark, and RAG tests.
- **106 passed** in ~17–18 seconds with 0 failures.

---

## 3. Verification Matrix Summary

| Test Category | Test File | Test Count | Status |
|:---|:---|:---:|:---:|
| **Database Connection & Masking** | `tests/test_database_connection.py` | 4 | [PASS] |
| **SQL Schema DDL & Invariants** | `tests/test_schema.py` | 1 | [PASS] |
| **Ingestion Generator** | `tests/test_ingestion_generator.py` | 5 | [PASS] |
| **Ingestion Validator** | `tests/test_ingestion_validator.py` | 9 | [PASS] |
| **Transactional Loader** | `tests/test_ingestion_loader.py` | 4 | [PASS] |
| **Document Chunking & Vector Storage** | `tests/test_embedding_pipeline.py` | 8 | [PASS] |
| **Embeddings & Model Registry** | `tests/test_embeddings.py` | 3 | [PASS] |
| **Exact Flat Semantic Retrieval** | `tests/test_exact_retrieval.py` | 7 | [PASS] |
| **In-Memory & Hybrid Retrieval** | `tests/test_retrieval.py` | 3 | [PASS] |
| **Index Experiments (HNSW & IVFFlat)** | `tests/test_index_experiments.py` | 7 | [PASS] |
| **Hybrid Relational-Vector Engine** | `tests/test_hybrid_retrieval.py` | 9 | [PASS] |
| **Multimodal Imaging & Cross-Modal** | `tests/test_multimodal_retrieval.py` | 14 | [PASS] |
| **Security Context & RLS Filters** | `tests/test_security.py` | 3 | [PASS] |
| **Live Database RBAC & Security Matrix** | `tests/test_security_authorization.py` | 13 | [PASS] |
| **Benchmarking Suite & Telemetry** | `tests/test_benchmark_suite.py` | 8 | [PASS] |
| **Clinical RAG & Decoupled Evaluator** | `tests/test_rag_pipeline.py` | 8 | [PASS] |
| **TOTAL VERIFIED TESTS** | **All 16 Test Suites** | **106** | **[100% PASS]** |

---

## 4. Troubleshooting Guide

| Issue / Error | Root Cause | Resolution Command |
|:---|:---|:---|
| `No module named 'sentence_transformers'` | PyTorch/SentenceTransformers not installed in `.venv`. | System automatically runs in deterministic offline encoder mode. To install PyTorch: `pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu` |
| `connection to server at "localhost", port 5432 failed` | PostgreSQL service is not started. | Run `Start-Service postgresql*` in PowerShell (Admin). |
| `psycopg.errors.InsufficientPrivilege` | User does not have access to table under active role. | Expected security behavior when testing least-privilege roles (e.g. `clinical_researcher`). |
| `ExternalAccessBlockedError` | External LLM called without opt-in. | Pass `--allow-external` or run with `--provider mock` / `--provider retrieval-only`. |
