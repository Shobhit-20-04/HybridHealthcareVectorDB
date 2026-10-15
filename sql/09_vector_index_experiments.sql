-- =============================================================================
-- Migration 09: Controlled Vector-Index Empirical Benchmark Scripts
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- Extension: PostgreSQL pgvector (v0.8.6+)
-- =============================================================================
-- This script provides standalone, repeatable, and parameterized SQL routines
-- for managing, tuning, and verifying vector indexes (HNSW, IVFFlat, Flat)
-- on clinical document chunk embeddings (384-dimensional dense vectors).
--
-- Index Types Evaluated:
--   1. Exact Flat Sequential Scan (Ground-truth baseline, zero index overhead, 100% recall)
--   2. HNSW (Hierarchical Navigable Small World graph, fast ANN, high recall, larger memory)
--   3. IVFFlat (Inverted File Flat Voronoi centroids, low memory, fast build, requires training data)
-- =============================================================================

-- =============================================================================
-- SECTION 1: Teardown Existing Vector Indexes
-- Ensures clean, unpolluted baseline state before running controlled benchmarks.
-- =============================================================================

DROP INDEX IF EXISTS idx_chunk_embeddings_hnsw_cosine;
DROP INDEX IF EXISTS idx_chunk_embeddings_ivfflat_cosine;

-- =============================================================================
-- SECTION 2: Exact Flat Search (Ground-Truth Baseline)
-- Disables approximate index scans and bitmap scans to force full table scans.
-- Every vector is computed with cosine distance (<=>), guaranteeing 1.0 Recall.
-- =============================================================================

-- Session knobs for exact brute-force search
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SET enable_seqscan = on;

-- Sample Ground-Truth Exact Search Query with Physical Plan Verification
-- Replace sample vector with target 384-dimensional query vector
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    dc.document_id,
    cd.title,
    cd.document_type,
    dc.chunk_text,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
JOIN clinical_documents cd ON dc.document_id = cd.document_id
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- Reset session knobs
RESET enable_indexscan;
RESET enable_bitmapscan;
RESET enable_seqscan;

-- =============================================================================
-- SECTION 3: HNSW (Hierarchical Navigable Small World) Index Lifecycle
-- Constructs a multi-layer graph where bottom layers contain all vectors and
-- top layers have sparse long-distance skip links.
--
-- Parameters:
--   m (default 16, range 4-64): Max number of bidirectional links per vector node.
--       Higher m increases graph density, memory size, and recall at the cost of build time.
--   ef_construction (default 64, range 16-512): Size of dynamic candidate list
--       during graph construction. Higher values improve graph quality and recall.
--   hnsw.ef_search (default 40, range 1-1000): Dynamic candidate list during queries.
--       Tuning ef_search at query time trades query latency for Recall@K without rebuilding.
-- =============================================================================

-- 3.1 Drop any conflicting vector indexes
DROP INDEX IF EXISTS idx_chunk_embeddings_ivfflat_cosine;
DROP INDEX IF EXISTS idx_chunk_embeddings_hnsw_cosine;

-- 3.2 Build HNSW Index
-- Recommended standard parameters for 384-dim text embeddings
CREATE INDEX idx_chunk_embeddings_hnsw_cosine
    ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 3.3 HNSW Query Parameter Sweeps (Tuning ef_search)
-- Test regime A: ef_search = 10 (Fastest latency, lower recall)
SET enable_seqscan = off;
SET enable_indexscan = on;
SET hnsw.ef_search = 10;

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- Test regime B: ef_search = 40 (Default balanced profile)
SET hnsw.ef_search = 40;

-- Test regime C: ef_search = 100 (Maximum recall approaching exact search)
SET hnsw.ef_search = 100;

-- Reset session knobs
RESET enable_seqscan;
RESET enable_indexscan;
RESET hnsw.ef_search;

-- =============================================================================
-- SECTION 4: IVFFlat (Inverted File Flat) Index Lifecycle
-- Partitions vector space into Voronoi cells using k-means clustering.
--
-- INVARIANT & PREREQUISITE:
--   IVFFlat MUST be created ONLY after sufficient vectors are present in the table.
--   Building IVFFlat on an empty or sparsely populated table causes empty or
--   degenerate Voronoi centroids, rendering the index ineffective.
--
-- Parameters:
--   lists (range 1-32768): Number of Voronoi centroids. Recommended: rows / 1000
--       for up to 1M rows, or sqrt(rows) for smaller datasets (e.g., lists = 25 for ~800 rows).
--   ivfflat.probes (default 1, range 1-lists): Number of nearest Voronoi lists to inspect.
--       Higher probes scan more centroids, trading latency for higher Recall@K.
-- =============================================================================

-- 4.1 Data Sufficiency Verification Guard
DO $$
DECLARE
    vector_count INT;
    desired_lists INT := 25;
BEGIN
    SELECT COUNT(*) INTO vector_count FROM chunk_embeddings WHERE model_id = 'sentence-transformers/all-MiniLM-L6-v2';
    IF vector_count < desired_lists THEN
        RAISE EXCEPTION 'Insufficient vectors to train IVFFlat index: found % rows, minimum required %', vector_count, desired_lists;
    END IF;
    RAISE NOTICE 'Sufficient vectors verified: % rows available for IVFFlat clustering.', vector_count;
END $$;

-- 4.2 Drop any conflicting vector indexes
DROP INDEX IF EXISTS idx_chunk_embeddings_hnsw_cosine;
DROP INDEX IF EXISTS idx_chunk_embeddings_ivfflat_cosine;

-- 4.3 Build IVFFlat Index
CREATE INDEX idx_chunk_embeddings_ivfflat_cosine
    ON chunk_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 25);

-- 4.4 IVFFlat Query Parameter Sweeps (Tuning ivfflat.probes)
-- Test regime A: probes = 1 (Fastest, scans only closest Voronoi cell, lower recall)
SET enable_seqscan = off;
SET enable_indexscan = on;
SET ivfflat.probes = 1;

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT 
    ce.chunk_id,
    (ce.embedding <=> '[0.01, 0.02, ...]'::vector) AS cosine_distance
FROM chunk_embeddings ce
WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
ORDER BY ce.embedding <=> '[0.01, 0.02, ...]'::vector ASC
LIMIT 10;

-- Test regime B: probes = 5 (Intermediate balance)
SET ivfflat.probes = 5;

-- Test regime C: probes = 10 (Higher recall, scans 40% of centroids for lists=25)
SET ivfflat.probes = 10;

-- Reset session knobs
RESET enable_seqscan;
RESET enable_indexscan;
RESET ivfflat.probes;

-- =============================================================================
-- SECTION 5: Index Footprint and Catalog Inspection
-- Queries pg_class and pg_indexes to report live disk size and index definitions.
-- =============================================================================

SELECT 
    i.indexname,
    i.indexdef,
    pg_size_pretty(pg_relation_size(i.indexname::regclass)) AS disk_size_pretty,
    pg_relation_size(i.indexname::regclass) AS disk_size_bytes
FROM pg_indexes i
WHERE i.tablename = 'chunk_embeddings'
ORDER BY disk_size_bytes DESC;

-- Re-establish default HNSW index for ongoing application use
DROP INDEX IF EXISTS idx_chunk_embeddings_ivfflat_cosine;
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw_cosine
    ON chunk_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
