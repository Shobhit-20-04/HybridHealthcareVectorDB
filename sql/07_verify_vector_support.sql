-- =============================================================================
-- Migration 07: Vector Support & Schema Verification Script
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================
-- Usage: psql -U postgres -d healthcare_vectordb -f sql/07_verify_vector_support.sql

-- -----------------------------------------------------------------------------
-- 1. Check Extension Registration & Version
-- -----------------------------------------------------------------------------
SELECT extname, extversion 
FROM pg_extension 
WHERE extname = 'vector';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'CRITICAL: pgvector extension is NOT installed in this database!';
    ELSE
        RAISE NOTICE 'SUCCESS: pgvector extension is active.';
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- 2. Verify Vector Mathematical Operators
-- -----------------------------------------------------------------------------
SELECT 
    '[1,0,0]'::vector <=> '[0,1,0]'::vector AS cosine_distance,
    '[1,0,0]'::vector <-> '[0,1,0]'::vector AS l2_distance,
    '[1,0,0]'::vector <#> '[1,0,0]'::vector AS negative_inner_product;

-- -----------------------------------------------------------------------------
-- 3. Verify All 9 Schema Tables
-- -----------------------------------------------------------------------------
SELECT 
    table_name,
    CASE 
        WHEN table_name IN (
            'patients', 'encounters', 'diagnoses', 'clinical_documents',
            'document_chunks', 'embedding_models', 'chunk_embeddings',
            'medical_images', 'image_embeddings'
        ) THEN 'VERIFIED'
        ELSE 'OTHER'
    END AS status
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- -----------------------------------------------------------------------------
-- 4. Verify Vector Column Dimensions
-- -----------------------------------------------------------------------------
SELECT 
    table_name, 
    column_name, 
    udt_name,
    atttypmod AS type_modifier
FROM information_schema.columns c
JOIN pg_attribute a ON a.attname = c.column_name
JOIN pg_class t ON t.relname = c.table_name
WHERE c.table_schema = 'public' 
  AND c.udt_name = 'vector'
  AND a.attrelid = t.oid;

-- -----------------------------------------------------------------------------
-- 5. Functional HNSW Index Test (Temporary Table)
-- -----------------------------------------------------------------------------
CREATE TEMP TABLE _test_vector_ann (
    id SERIAL PRIMARY KEY,
    label TEXT,
    embedding vector(3)
);

INSERT INTO _test_vector_ann (label, embedding) VALUES
    ('X_Axis', '[1, 0, 0]'),
    ('Y_Axis', '[0, 1, 0]'),
    ('Z_Axis', '[0, 0, 1]');

CREATE INDEX _test_hnsw_idx ON _test_vector_ann USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Execute ANN search
SELECT 
    id, 
    label, 
    embedding <=> '[0.9, 0.1, 0]'::vector AS cosine_dist
FROM _test_vector_ann
ORDER BY embedding <=> '[0.9, 0.1, 0]'::vector
LIMIT 1;

DROP TABLE _test_vector_ann;
