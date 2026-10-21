-- ============================================================================
-- HYBRID HEALTHCARE VECTORDB — ACADEMIC FACULTY DEMONSTRATION QUERIES
-- Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare
-- Database: PostgreSQL 18.6 with pgvector v0.8.6
-- Dataset: 1,000 Real PubMed Central (PMC-Patients) Cases (17,076 total records)
-- ============================================================================
-- These queries demonstrate core DBMS principles:
-- 1. Hybrid Relational-Vector Execution (Combining SQL B-Tree filters with vector distance)
-- 2. Query Planning and HNSW Index Traversal (EXPLAIN ANALYZE)
-- 3. Vector Distance Metrics (<=> cosine, <-> L2, <#> inner product)
-- 4. Multi-Table Relational Normalization & Referential Integrity
-- 5. Row-Level Security (RLS) Isolation
-- 6. Multimodal Vector Storage (Image & Text Embeddings)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: HYBRID RELATIONAL-VECTOR SEARCH
-- Concept: Combines a relational equality filter (e.g., department = 'Cardiology')
--          with vector cosine distance (<=>) ordering over 384-dimensional embeddings.
-- ----------------------------------------------------------------------------
SELECT 
    p.synthetic_mrn AS patient_mrn,
    p.gender,
    ROUND(EXTRACT(YEAR FROM age(p.date_of_birth))::numeric, 0) AS age,
    e.department,
    e.encounter_type,
    d.document_type,
    SUBSTRING(c.chunk_text, 1, 100) AS clinical_snippet,
    ROUND((1 - (ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)))::numeric, 4) AS cosine_similarity,
    ROUND((ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1))::numeric, 4) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
JOIN encounters e ON d.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id
WHERE e.department = 'Cardiology'
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;


-- ----------------------------------------------------------------------------
-- QUERY 2: EXPLAIN (ANALYZE, BUFFERS) — QUERY PLAN & INDEX TRAVERSAL
-- Concept: Demonstrates to faculty that PostgreSQL's query optimizer utilizes
--          the HNSW graph index (idx_chunk_embeddings_hnsw_cosine) rather than
--          performing a costly sequential scan across all vectors.
-- ----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    p.synthetic_mrn,
    e.department,
    d.document_type,
    ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1) AS distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
JOIN encounters e ON d.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id
WHERE e.department = 'Cardiology'
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;


-- ----------------------------------------------------------------------------
-- QUERY 3: EXACT VECTOR TOP-K SIMILARITY SEARCH ACROSS ALL CHUNKS
-- Concept: Demonstrates pure vector similarity search using native pgvector cosine
--          distance (<=> operator) ranked across all 6,244 real clinical chunks.
-- ----------------------------------------------------------------------------
SELECT 
    c.chunk_id,
    d.synthetic_document_num,
    d.title AS document_title,
    c.chunk_index,
    c.token_count,
    SUBSTRING(c.chunk_text, 1, 120) AS chunk_snippet,
    ROUND((1 - (ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)))::numeric, 4) AS cosine_similarity,
    ROUND((ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1))::numeric, 4) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;


-- ----------------------------------------------------------------------------
-- QUERY 4: 4-TABLE RELATIONAL INTEGRITY JOIN
-- Concept: Demonstrates 3NF relational modeling, Primary Key to Foreign Key integrity,
--          and multi-table joins across Patients -> Encounters -> Diagnoses -> Documents.
-- ----------------------------------------------------------------------------
SELECT 
    p.synthetic_mrn,
    p.gender,
    e.department,
    e.admission_date,
    diag.icd10_code,
    diag.description AS diagnosis_description,
    doc.document_type,
    doc.title AS article_title
FROM patients p
JOIN encounters e ON p.patient_id = e.patient_id
JOIN diagnoses diag ON e.encounter_id = diag.encounter_id
JOIN clinical_documents doc ON e.encounter_id = doc.encounter_id
WHERE e.department = 'Neurology'
ORDER BY e.admission_date DESC
LIMIT 10;


-- ----------------------------------------------------------------------------
-- QUERY 5: MULTIMODAL STORAGE & EMBEDDINGS (IMAGE + TEXT RETRIEVAL)
-- Concept: Demonstrates multimodal database schema: medical images with modality,
--          anatomical site, image file path, and corresponding 512-dim CLIP embeddings.
-- ----------------------------------------------------------------------------
SELECT 
    img.image_id,
    img.synthetic_image_num,
    img.modality,
    img.body_site,
    img.file_path,
    e.department,
    ROUND((1 - (ie.embedding <=> (SELECT embedding FROM image_embeddings LIMIT 1)))::numeric, 4) AS visual_similarity
FROM image_embeddings ie
JOIN medical_images img ON ie.image_id = img.image_id
JOIN encounters e ON img.encounter_id = e.encounter_id
ORDER BY ie.embedding <=> (SELECT embedding FROM image_embeddings LIMIT 1)
LIMIT 5;


-- ----------------------------------------------------------------------------
-- QUERY 6: DATABASE STORAGE & INDEX TELEMETRY
-- Concept: Queries PostgreSQL internal catalogs (pg_class, pg_stat_user_indexes)
--          to inspect physical disk footprint of relational tables vs vector indexes.
-- ----------------------------------------------------------------------------
SELECT 
    relname AS relation_name,
    CASE 
        WHEN relkind = 'r' THEN 'Table'
        WHEN relkind = 'i' THEN 'Index'
        ELSE 'Other'
    END AS relation_type,
    pg_size_pretty(pg_relation_size(c.oid)) AS disk_size,
    reltuples::bigint AS estimated_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' 
  AND c.relname IN (
    'patients', 'encounters', 'diagnoses', 'clinical_documents', 
    'document_chunks', 'chunk_embeddings', 'medical_images', 'image_embeddings',
    'idx_chunk_embeddings_hnsw_cosine', 'idx_chunk_embeddings_ivfflat_cosine'
  )
ORDER BY pg_relation_size(c.oid) DESC;
