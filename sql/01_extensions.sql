-- =============================================================================
-- Migration 01: PostgreSQL Extensions Setup
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

-- Ensure vector extension is installed and activated
CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure cryptographic / UUID generation extension is activated
-- PostgreSQL 13+ includes gen_random_uuid() natively, but pgcrypto / uuid-ossp
-- guarantee backwards compatibility for UUID v4 operations.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify extension registration
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension is not installed in the PostgreSQL instance.';
    END IF;
    RAISE NOTICE 'pgvector and UUID extensions verified successfully.';
END $$;
