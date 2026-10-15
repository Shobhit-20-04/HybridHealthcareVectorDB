-- =============================================================================
-- Migration / Script 08: Ground-Truth Exact Semantic Similarity Search Baseline
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================
--
-- PURPOSE:
--   Establishes the authoritative ground-truth baseline for nearest-neighbor 
--   retrieval without approximate vector index structures (HNSW or IVFFlat).
--   Used to calculate Recall@K, latency trade-offs, and ranking fidelity.
--
-- MATHEMATICAL FORMULATION:
--   - Cosine Distance:   D_C(u, v) = 1.0 - (u . v) / (||u||_2 * ||v||_2)  in [0, 2]
--   - Cosine Similarity: S_C(u, v) = 1.0 - D_C(u, v)                       in [-1, 1]
--
--   Note:
--     * pgvector's <=> operator calculates Cosine Distance (0 = identical, 2 = opposite).
--     * Smaller distance indicates closer semantic proximity (higher similarity).
--     * ORDER BY embedding <=> query_vector ASC retrieves the highest semantic matches.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Session Tuning to Guarantee Exact Sequential Flat Scan (No Index)
-- -----------------------------------------------------------------------------
-- Disabling indexscan and bitmapscan forces PostgreSQL's cost planner to perform
-- an exhaustive Sequential Scan across all tuples in chunk_embeddings.
BEGIN;

SET LOCAL enable_indexscan = off;
SET LOCAL enable_bitmapscan = off;

-- -----------------------------------------------------------------------------
-- 2. Parameterized Exact Search Query Formulation
-- -----------------------------------------------------------------------------
-- Joins the vector storage layer back to the normalized clinical relational entities:
--   chunk_embeddings -> document_chunks -> clinical_documents -> encounters -> patients
--
-- Replace ':query_vector' with a 384-dimensional vector literal, e.g. '[0.012, -0.045, ...]'
-- -----------------------------------------------------------------------------

SELECT 
    ROW_NUMBER() OVER (ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384) ASC) AS rank,
    ce.embedding_id,
    ce.chunk_id,
    dc.document_id,
    cd.synthetic_document_num,
    p.synthetic_mrn,
    cd.patient_id,
    cd.encounter_id,
    e.department,
    cd.document_type,
    cd.title,
    dc.chunk_index,
    dc.token_count,
    dc.char_length,
    dc.chunk_text AS snippet,
    -- Cosine Distance (pgvector native metric, lower is closer)
    (ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384)) AS cosine_distance,
    -- Cosine Similarity (higher is closer)
    (1.0 - (ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384))) AS cosine_similarity,
    cd.metadata AS doc_metadata
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON cd.patient_id = p.patient_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384) ASC
LIMIT 5;

COMMIT;

-- -----------------------------------------------------------------------------
-- 3. Execution Plan Inspection (EXPLAIN ANALYZE)
-- -----------------------------------------------------------------------------
-- Demonstrates that PostgreSQL executes a pure Seq Scan on chunk_embeddings
-- and a top-N heapsort over the calculated cosine distances.
-- -----------------------------------------------------------------------------
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    ce.chunk_id,
    (ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384)) AS cosine_distance
FROM chunk_embeddings ce
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> '[0.01, 0.02, 0.03]'::vector(384) ASC
LIMIT 5;
