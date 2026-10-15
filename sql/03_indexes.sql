-- =============================================================================
-- Migration 03: Relational B-Tree, JSONB GIN, and pgvector Indexes
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

-- =============================================================================
-- 1. Foreign Key Performance Indexes (B-Tree)
-- Prevents table locks and sequential scans during relational join execution
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_encounters_patient_id 
    ON encounters (patient_id);

CREATE INDEX IF NOT EXISTS idx_diagnoses_encounter_id 
    ON diagnoses (encounter_id);

CREATE INDEX IF NOT EXISTS idx_diagnoses_patient_id 
    ON diagnoses (patient_id);

CREATE INDEX IF NOT EXISTS idx_clinical_docs_encounter_id 
    ON clinical_documents (encounter_id);

CREATE INDEX IF NOT EXISTS idx_clinical_docs_patient_id 
    ON clinical_documents (patient_id);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_document_id 
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk_id 
    ON chunk_embeddings (chunk_id);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_id 
    ON chunk_embeddings (model_id);

CREATE INDEX IF NOT EXISTS idx_medical_images_patient_id 
    ON medical_images (patient_id);

CREATE INDEX IF NOT EXISTS idx_medical_images_encounter_id 
    ON medical_images (encounter_id);

CREATE INDEX IF NOT EXISTS idx_medical_images_document_id 
    ON medical_images (document_id);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_image_id 
    ON image_embeddings (image_id);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_model_id 
    ON image_embeddings (model_id);

-- =============================================================================
-- 2. Relational Metadata Filtering Indexes (B-Tree)
-- Optimized for high-selectivity predicates in hybrid SQL queries
-- =============================================================================

-- Composite index for clinical department & encounter type filtering
CREATE INDEX IF NOT EXISTS idx_encounters_dept_type 
    ON encounters (department, encounter_type);

-- Temporal range index for longitudinal clinical encounters
CREATE INDEX IF NOT EXISTS idx_encounters_admission_date 
    ON encounters (admission_date);

-- Composite index for ICD-10 disease cohort selection
CREATE INDEX IF NOT EXISTS idx_diagnoses_icd10_code 
    ON diagnoses (icd10_code, patient_id);

-- Document categorization and date filtering
CREATE INDEX IF NOT EXISTS idx_clinical_docs_type_date 
    ON clinical_documents (document_type, created_at DESC);

-- Demographic stratification index (age/gender cohorts)
CREATE INDEX IF NOT EXISTS idx_patients_gender_dob 
    ON patients (gender, date_of_birth);

-- Imaging modality and anatomical body site index
CREATE INDEX IF NOT EXISTS idx_medical_images_modality_site 
    ON medical_images (modality, body_site);

-- =============================================================================
-- 3. Semistructured Metadata Indexes (GIN)
-- Accelerates JSONB path-based queries on clinical and imaging metadata
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_clinical_docs_metadata_gin 
    ON clinical_documents USING GIN (metadata jsonb_path_ops);

CREATE INDEX IF NOT EXISTS idx_medical_images_metadata_gin 
    ON medical_images USING GIN (metadata jsonb_path_ops);

-- =============================================================================
-- 4. Vector Approximate Nearest Neighbor (ANN) Indexes
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 4.1 HNSW (Hierarchical Navigable Small World) Index on Text Chunk Embeddings
-- Optimized for high recall and ultra-low query latency under concurrent reads.
-- Uses vector_cosine_ops for cosine similarity (<=>).
-- Parameters:
--   m = 16: Max number of bidirectional links per vector node.
--   ef_construction = 64: Size of dynamic candidate list during graph building.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw_cosine
    ON chunk_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- 4.2 HNSW Index on Multimodal Image Embeddings (512-dim)
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_image_embeddings_hnsw_cosine
    ON image_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- NOTE ON BENCHMARKING ALTERNATIVES (IVFFlat & Flat Sequential Scan):
-- For IVFFlat index experiments:
--   CREATE INDEX idx_chunk_embeddings_ivfflat_cosine
--   ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 100);
--   (Requires training vectors already present in the table before creation)
--
-- For Exact Flat search experiments:
--   Vector indexes are omitted or SET enable_indexscan = off; SET enable_bitmapscan = off;
-- =============================================================================
