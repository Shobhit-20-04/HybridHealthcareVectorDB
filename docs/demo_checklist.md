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

### Step 4: Ingest Real-World PMC Clinical Data (1,000 Patient Cases)
```powershell
# OPTION A (RECOMMENDED): 1,000 Real Clinical Cases from PubMed Central
python scripts/download_real_pmc_data.py --num-cases 1000
python scripts/ingest_real_pmc_data.py --mode replace

# OPTION B: Deterministic Synthetic Data (Offline Testing)
# python scripts/generate_data.py --preset small --seed 42
# python scripts/load_data.py --truncate
# python scripts/embed_documents.py --smoke-test
# python scripts/embed_images.py --smoke-test
```
*Expected Result (Option A):*
- Downloads & curates 1,000 genuine clinical cases from PubMed Central (`PMC-Patients`).
- Transactionally loads 1,000 patients, 1,000 encounters, 1,000 diagnoses, 1,000 documents, 6,244 chunks, 292 medical images.
- Generates 6,244 text embeddings (`vector(384)`) and 292 vision embeddings (`vector(512)`).
- Total records: **17,076 rows** in PostgreSQL.

---

### Step 5: Verify Database Ingestion & Referential Integrity
```powershell
python scripts/validate_ingestion.py
```
*Expected Result:*
- Checks for orphan records across all 8 foreign key relationships (0 orphans found).
- Checks chunk sequences, dates, and provenance hashes.
- Outputs `[V] ALL INGESTION INTEGRITY AND REFERENTIAL CHECKS PASSED.`

---

### Step 6: Audit Live Database Table Counts
```powershell
python scripts/count_records.py
```
*Expected Result:*
- Displays synchronized table audit: 1,000 patients, 1,000 encounters, 1,000 diagnoses, 1,000 documents, 6,244 chunks, 6,244 chunk embeddings, 292 images, 292 image embeddings (17,076 total rows).

---

### Step 7: Run Exact Semantic Search (Ground-Truth Baseline)
```powershell
python scripts/search_exact.py "acute myocardial infarction with ST elevation" --top-k 3
```
*Expected Result:*
- Executes brute-force sequential flat scan (`enable_indexscan=off`).
- Returns top-3 ranked chunks with cosine distances (lower is closer), snippets, titles, and timing breakdown (~15–75 ms).

---

### Step 8: Run Hybrid Relational–Vector Search
```powershell
python scripts/search_hybrid.py "pulmonary infiltrate dyspnea" --department Pulmonology --top-k 3
```
*Expected Result:*
- Joins patient demographics (MRN, Age, Gender, DOB), encounter details (Department, Admission Date), and ICD-10 diagnostic codes.
- Restricts vector similarity calculation strictly to `department = 'Pulmonology'`.
- Returns top-3 ranked clinical passages with complete relational metadata.

---

### Step 9: Verify Database Security, Roles & Row-Level Security (RLS)
```powershell
python scripts/verify_security.py
```
*Expected Result:*
- Audits PostgreSQL catalog for roles and RLS flags.
- Executes 9/9 live authorization checks: Admin visibility, Physician departmental scoping, Cross-dept isolation, Vector search leakage prevention, Radiologist imaging access, Non-imaging block, Researcher de-identified view access, Direct table access denials.
- Outputs `[V] ALL 9 AUTHORIZATION & SECURITY TESTS PASSED SUCCESSFULLY!`

---

### Step 10: Run Command-Line Clinical RAG Demonstration
```powershell
python scripts/run_rag_demo.py "sarcoidosis and myocardial infarction" --evaluate
```
*Expected Result:*
- Retrieves top-k relevant passages from PostgreSQL.
- Deterministic local mock LLM synthesizes evidence-grounded answer citing document IDs (e.g. `[PMC-DOC-...]`).
- Decoupled evaluation verifies 100% Citation Precision, 100% Claim Groundedness, and Correct Abstention when evidence is absent.

---

### Step 11: Run Full Automated Test Suite (106 Tests)
```powershell
pytest tests/ -v
```
*Expected Result:*
- `106 passed in ~22s` with 100% passing tests across unit, integration, retrieval, security, multimodal, and benchmark suites.

---

### Step 12: Run Benchmark Suite Smoke Test
```powershell
python scripts/run_benchmarks.py --smoke-test
```
*Expected Result:*
- Builds HNSW (12 MB) and IVFFlat (10 MB) indexes on the 6,244 vectors.
- Compares Exact Flat vs HNSW vs IVFFlat across 4 selectivity tiers.
- Outputs markdown table, JSON report, and CSV summaries.

---

### Step 13: Launch Interactive React Web Application
```powershell
python scripts/run_app.py
```
*Expected Result:*
- Automatically loads the built React production bundle and launches the FastAPI backend.
- Opens at `http://127.0.0.1:8000`.
- Provides an intuitive, accessible visual interface for demonstrating:
  1. **Hybrid Search Explorer** (natural language queries, department/age/document-type filters, EXPLAIN execution plan toggle).
  2. **Multimodal Search** (Text-to-Image and Image-to-Image similarity search with visual phantom renderings).
  3. **Row-Level Security (RLS) Simulator** (switch roles between Admin, Physician, Radiologist, Researcher and inspect real-time authorization results).
  4. **Clinical RAG Assistant** (ask clinical questions, verify document citations, and view decoupled evaluation scores).
  5. **Database Architecture & Telemetry** (live table row counts, relation disk sizes, and index configurations).
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
