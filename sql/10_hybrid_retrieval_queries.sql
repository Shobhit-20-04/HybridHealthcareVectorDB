-- =============================================================================
-- Migration 10: Hybrid Relational–Vector Query Execution & Selectivity Benchmarks
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- Extension: PostgreSQL pgvector (v0.8.6+)
-- =============================================================================
-- This script provides standalone, parameterized SQL queries demonstrating
-- hybrid relational filtering combined with dense vector similarity ranking.
--
-- Features Demonstrated:
--   1. Parameterized metadata filtering (department, document_type, age ranges, ICD-10 diagnoses)
--   2. Multi-table relational joins connecting chunk vectors to complete clinical metadata
--   3. Execution plan inspection via EXPLAIN (ANALYZE, BUFFERS) across 4 selectivity tiers
--   4. Optimizer behavior comparison: Relational B-Tree Pre-filtering vs Vector Iterative Scan
-- =============================================================================

-- =============================================================================
-- SECTION 1: Baseline Join & Full Metadata Extraction Template
-- Connects vector chunk to complete longitudinal clinical context.
-- =============================================================================

-- Query Template: Extracts full joined clinical record with cosine distance ranking
-- Note: Replace query vector placeholder with 384-dimensional dense vector
SELECT 
    ce.chunk_id,
    dc.chunk_index,
    dc.chunk_text,
    dc.token_count,
    cd.document_id,
    cd.synthetic_document_num,
    cd.title AS document_title,
    cd.document_type,
    p.patient_id,
    p.synthetic_mrn,
    p.gender,
    p.date_of_birth,
    EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth))::int AS patient_age_at_encounter,
    e.encounter_id,
    e.department,
    e.encounter_type,
    e.admission_date,
    STRING_AGG(DISTINCT diag.icd10_code || ': ' || diag.diagnosis_description, '; ') AS encounter_diagnoses,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance,
    (1.0 - (ce.embedding <=> '[0.01, 0.02, ...]'::vector)) AS cosine_similarity
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
LEFT JOIN diagnoses diag ON e.encounter_id = diag.encounter_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
GROUP BY 
    ce.chunk_id, dc.chunk_index, dc.chunk_text, dc.token_count,
    cd.document_id, cd.synthetic_document_num, cd.title, cd.document_type,
    p.patient_id, p.synthetic_mrn, p.gender, p.date_of_birth,
    e.encounter_id, e.department, e.encounter_type, e.admission_date, ce.embedding
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- =============================================================================
-- SECTION 2: Filter Selectivity Tiers & Execution Plan Inspection
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Tier 1: Low Selectivity / Broad Filter (~60% - 80% matching rows)
-- Filter: Adult patients (age >= 18) in common clinical departments
-- Expected Plan: Optimizer utilizes HNSW vector index scan; surviving rows filtered post-index.
-- -----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    cd.title,
    cd.document_type,
    e.department,
    EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth))::int AS age,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS dist
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  AND EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) >= 18
  AND e.department IN ('Cardiology', 'Internal Medicine', 'Pulmonology', 'Emergency')
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- Tier 2: Moderate Selectivity (~15% - 30% matching rows)
-- Filter: Department = 'Cardiology' with specific encounter type
-- -----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    cd.title,
    cd.document_type,
    e.department,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS dist
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  AND e.department = 'Cardiology'
  AND e.encounter_type = 'inpatient'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- Tier 3: High Selectivity / Restrictive Filter (~1% - 5% matching rows)
-- Filter: Pulmonology department AND radiology reports AND specific age cohort
-- Expected Plan: B-Tree Bitmap Index Scan on idx_clinical_docs_type_date & idx_encounters_dept_type
-- Relational pre-filtering executes first, vector distances computed only on survivors!
-- -----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    cd.title,
    cd.document_type,
    e.department,
    p.synthetic_mrn,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS dist
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  AND e.department = 'Pulmonology'
  AND cd.document_type = 'radiology_report'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- Tier 4: Zero-Match / Ultra-Restrictive Filter (0% matching rows)
-- Filter: Impossible age range (age > 150) or disjoint condition
-- Expected Plan: Returns empty set cleanly with 0 rows and minimal buffer overhead.
-- -----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    cd.title,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS dist
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  AND EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth)) > 150
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- =============================================================================
-- SECTION 3: Explicit Pre-Filter (Subquery) vs Integrated Joint Hybrid
-- =============================================================================

-- Strategy A: Explicit Subquery / CTE Attribute-First Pre-Filtering
-- Restricts candidate space in CTE before vector similarity evaluation
WITH pre_filtered_candidates AS (
    SELECT 
        ce.chunk_id,
        ce.embedding,
        cd.title,
        cd.document_type,
        e.department
    FROM chunk_embeddings ce
    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
    JOIN clinical_documents cd ON dc.document_id = cd.document_id
    JOIN encounters e ON cd.encounter_id = e.encounter_id
    WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
      AND e.department = 'Cardiology'
      AND cd.document_type = 'discharge_summary'
)
SELECT 
    chunk_id,
    title,
    document_type,
    department,
    (embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance
FROM pre_filtered_candidates
ORDER BY embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- Strategy B: Approximate Index with Iterative Scan Control
-- Tune hnsw.iterative_scan to investigate candidate exhaust and result deficit
SET LOCAL enable_seqscan = off;
SET LOCAL enable_indexscan = on;
SET LOCAL hnsw.ef_search = 10;
SET LOCAL hnsw.iterative_scan = off; -- 'off' vs 'relaxed_order' vs 'strict_order'

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    cd.title,
    e.department,
    cd.document_type,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  AND e.department = 'Neurology'
  AND cd.document_type = 'pathology_report'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

RESET enable_seqscan;
RESET enable_indexscan;
RESET hnsw.ef_search;
RESET hnsw.iterative_scan;
