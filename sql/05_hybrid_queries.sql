-- =============================================================================
-- Migration 05: Benchmark Hybrid SQL Queries
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

-- =============================================================================
-- Query 1: Pure Exact Vector Similarity Search (Brute-Force Sequential Scan)
-- Used as the ground-truth baseline for calculating Recall@K
-- =============================================================================
-- SET LOCAL enable_indexscan = off;
-- SET LOCAL enable_bitmapscan = off;
SELECT 
    ce.embedding_id,
    ce.chunk_id,
    dc.chunk_text,
    cd.title AS document_title,
    cd.document_type,
    (ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector
LIMIT 10;

-- =============================================================================
-- Query 2: Approximate Nearest Neighbor Search via HNSW
-- =============================================================================
-- SET LOCAL enable_indexscan = on;
-- SET LOCAL hnsw.ef_search = 40;
SELECT 
    ce.embedding_id,
    ce.chunk_id,
    dc.chunk_text,
    cd.title AS document_title,
    cd.document_type,
    (ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector
LIMIT 10;

-- =============================================================================
-- Query 3: Hybrid Strategy 1 - Attribute-First (Pre-Filtering)
-- Filters encounters and documents by relational predicates before vector ranking
-- =============================================================================
WITH filtered_candidates AS (
    SELECT 
        ce.embedding_id,
        ce.chunk_id,
        ce.embedding,
        dc.chunk_text,
        cd.title AS doc_title,
        cd.document_type,
        e.department,
        e.encounter_type,
        e.admission_date,
        p.synthetic_mrn
    FROM chunk_embeddings ce
    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
    JOIN clinical_documents cd ON dc.document_id = cd.document_id
    JOIN encounters e ON cd.encounter_id = e.encounter_id
    JOIN patients p ON e.patient_id = p.patient_id
    WHERE e.department = 'Cardiology'
      AND e.encounter_type = 'inpatient'
)
SELECT 
    embedding_id,
    chunk_id,
    chunk_text,
    doc_title,
    department,
    synthetic_mrn,
    (embedding <=> '[0.01, 0.02, 0.03...]'::vector) AS cosine_distance
FROM filtered_candidates
ORDER BY embedding <=> '[0.01, 0.02, 0.03...]'::vector
LIMIT 10;

-- =============================================================================
-- Query 4: Hybrid Strategy 2 - Vector-First (Post-Filtering)
-- Retrieves top-M nearest vectors across database, then prunes with relational filters
-- =============================================================================
WITH nearest_vectors AS (
    SELECT 
        ce.embedding_id,
        ce.chunk_id,
        (ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector) AS cosine_distance
    FROM chunk_embeddings ce
    ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03...]'::vector
    LIMIT 100
)
SELECT 
    nv.embedding_id,
    dc.chunk_id,
    dc.chunk_text,
    cd.title AS doc_title,
    e.department,
    e.encounter_type,
    p.synthetic_mrn,
    nv.cosine_distance
FROM nearest_vectors nv
JOIN document_chunks dc ON nv.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id
WHERE e.department = 'Cardiology'
  AND e.encounter_type = 'inpatient'
ORDER BY nv.cosine_distance
LIMIT 10;

-- =============================================================================
-- Query 5: Cross-Modal Medical Image Retrieval (Query Text -> Image Embeddings)
-- =============================================================================
SELECT 
    mi.image_id,
    mi.synthetic_image_num,
    mi.modality,
    mi.body_site,
    mi.synthetic_file_path,
    (ie.embedding <=> '[0.01, 0.02, 0.03...]'::vector) AS multimodal_cosine_distance
FROM image_embeddings ie
JOIN medical_images mi ON ie.image_id = mi.image_id
WHERE mi.modality = 'CR' AND mi.body_site = 'Chest'
ORDER BY ie.embedding <=> '[0.01, 0.02, 0.03...]'::vector
LIMIT 5;
