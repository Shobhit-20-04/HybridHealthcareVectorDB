-- =============================================================================
-- Master Migration Runner (Repeatable)
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================
-- Executes all approved migration scripts in numerical sequence.
-- Usage:
--   psql -U postgres -d healthcare_vectordb -f sql/init_all.sql

\echo '-----------------------------------------------------------------------------'
\echo '  APPLYING MIGRATION 01: Extensions Setup'
\echo '-----------------------------------------------------------------------------'
\i sql/01_extensions.sql

\echo '-----------------------------------------------------------------------------'
\echo '  APPLYING MIGRATION 02: Relational Schema & Vector Tables DDL'
\echo '-----------------------------------------------------------------------------'
\i sql/02_schema.sql

\echo '-----------------------------------------------------------------------------'
\echo '  APPLYING MIGRATION 03: Relational B-Tree, GIN & Vector HNSW Indexes'
\echo '-----------------------------------------------------------------------------'
\i sql/03_indexes.sql

\echo '-----------------------------------------------------------------------------'
\echo '  APPLYING MIGRATION 04: Seed Canonical Embedding Models'
\echo '-----------------------------------------------------------------------------'
\i sql/04_seed_embedding_models.sql

\echo '-----------------------------------------------------------------------------'
\echo '  APPLYING MIGRATION 06: Row-Level Security (RLS) & Role Policies'
\echo '-----------------------------------------------------------------------------'
\i sql/06_security_rls.sql

\echo '-----------------------------------------------------------------------------'
\echo '  RUNNING VERIFICATION SCRIPT 07: Vector Support & Schema Checks'
\echo '-----------------------------------------------------------------------------'
\i sql/07_verify_vector_support.sql
