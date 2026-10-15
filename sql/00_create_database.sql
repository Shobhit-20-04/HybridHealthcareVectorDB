-- =============================================================================
-- Migration 00: Database Creation Template (Repeatable / Idempotent)
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================
-- Run this script as a PostgreSQL superuser (e.g. postgres) connected to 'postgres'
-- psql -U postgres -d postgres -f sql/00_create_database.sql

-- Check if database exists; create via createdb or psql:
-- Shell command: createdb -U postgres healthcare_vectordb

SELECT 'CREATE DATABASE healthcare_vectordb WITH OWNER postgres ENCODING ''UTF8'''
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = 'healthcare_vectordb'
);

COMMENT ON DATABASE healthcare_vectordb IS 'Dedicated database for Hybrid Relational-Vector Healthcare Information Retrieval project.';
