-- =============================================================================
-- Migration 02: Relational Schema & Vector Storage DDL
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

-- Clean teardown (for deterministic schema resets in test/benchmark environments)
DROP TABLE IF EXISTS image_embeddings CASCADE;
DROP TABLE IF EXISTS medical_images CASCADE;
DROP TABLE IF EXISTS chunk_embeddings CASCADE;
DROP TABLE IF EXISTS embedding_models CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS clinical_documents CASCADE;
DROP TABLE IF EXISTS diagnoses CASCADE;
DROP TABLE IF EXISTS encounters CASCADE;
DROP TABLE IF EXISTS patients CASCADE;

-- -----------------------------------------------------------------------------
-- 1. Synthetic Patients
-- -----------------------------------------------------------------------------
CREATE TABLE patients (
    patient_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    synthetic_mrn VARCHAR(32) NOT NULL UNIQUE,
    gender VARCHAR(16) NOT NULL,
    date_of_birth DATE NOT NULL,
    blood_type VARCHAR(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_patient_gender CHECK (gender IN ('M', 'F', 'OTHER', 'UNKNOWN')),
    CONSTRAINT chk_patient_blood_type CHECK (blood_type IS NULL OR blood_type IN ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'))
);

COMMENT ON TABLE patients IS 'Synthetic demographic records for patient cohort analysis without PHI.';
COMMENT ON COLUMN patients.synthetic_mrn IS 'Deterministic synthetic medical record number (e.g. SYN-MRN-XXXXXX).';

-- -----------------------------------------------------------------------------
-- 2. Clinical Encounters
-- -----------------------------------------------------------------------------
CREATE TABLE encounters (
    encounter_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    synthetic_encounter_num VARCHAR(32) NOT NULL UNIQUE,
    encounter_type VARCHAR(32) NOT NULL,
    department VARCHAR(64) NOT NULL,
    admission_date TIMESTAMPTZ NOT NULL,
    discharge_date TIMESTAMPTZ,
    attending_clinician_id VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_encounter_type CHECK (encounter_type IN ('inpatient', 'outpatient', 'emergency', 'telehealth')),
    CONSTRAINT chk_discharge_after_admission CHECK (discharge_date IS NULL OR discharge_date >= admission_date)
);

COMMENT ON TABLE encounters IS 'Clinical hospital visits providing relational predicates for hybrid querying.';

-- -----------------------------------------------------------------------------
-- 3. Diagnoses (ICD-10)
-- -----------------------------------------------------------------------------
CREATE TABLE diagnoses (
    diagnosis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    patient_id UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    icd10_code VARCHAR(16) NOT NULL,
    diagnosis_description VARCHAR(255) NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    diagnosed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE diagnoses IS 'Structured relational ICD-10 diagnostic entries associated with encounters.';

-- -----------------------------------------------------------------------------
-- 4. Clinical Documents (Narratives & Semistructured Metadata)
-- -----------------------------------------------------------------------------
CREATE TABLE clinical_documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    patient_id UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    synthetic_document_num VARCHAR(32) NOT NULL UNIQUE,
    document_type VARCHAR(64) NOT NULL,
    author_id VARCHAR(32) NOT NULL,
    title VARCHAR(255) NOT NULL,
    full_text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(16) NOT NULL DEFAULT 'signed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_doc_type CHECK (document_type IN ('discharge_summary', 'progress_note', 'radiology_report', 'pathology_report', 'consult_note')),
    CONSTRAINT chk_doc_status CHECK (status IN ('draft', 'preliminary', 'signed', 'amended'))
);

COMMENT ON TABLE clinical_documents IS 'Full-text clinical narratives decoupled from high-dimensional vector embeddings.';
COMMENT ON COLUMN clinical_documents.metadata IS 'Semistructured attributes (e.g. extracted vitals, section boundaries, clinical tags).';

-- -----------------------------------------------------------------------------
-- 5. Document Chunks (Token-Segmented Units for Embeddings)
-- -----------------------------------------------------------------------------
CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES clinical_documents(document_id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INT NOT NULL,
    char_length INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT chk_chunk_token_count CHECK (token_count > 0),
    CONSTRAINT chk_chunk_char_length CHECK (char_length > 0),
    CONSTRAINT uq_document_chunk_idx UNIQUE (document_id, chunk_index)
);

COMMENT ON TABLE document_chunks IS 'Passages extracted from clinical documents conforming to transformer token windows.';

-- -----------------------------------------------------------------------------
-- 6. Embedding Models Registry
-- -----------------------------------------------------------------------------
CREATE TABLE embedding_models (
    model_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    modality VARCHAR(16) NOT NULL,
    dimension INT NOT NULL,
    distance_metric VARCHAR(32) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_model_modality CHECK (modality IN ('text', 'image', 'multimodal')),
    CONSTRAINT chk_model_dimension CHECK (dimension > 0),
    CONSTRAINT chk_model_metric CHECK (distance_metric IN ('cosine', 'l2', 'inner_product'))
);

COMMENT ON TABLE embedding_models IS 'Explicit registry of allowed embedding architectures, dimensions, and metrics.';

-- -----------------------------------------------------------------------------
-- 7. Document Chunk Embeddings (pgvector Dense Vector Storage)
-- -----------------------------------------------------------------------------
CREATE TABLE chunk_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    model_id VARCHAR(64) NOT NULL REFERENCES embedding_models(model_id) ON DELETE RESTRICT,
    dimension INT NOT NULL,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_chunk_embedding_dim CHECK (dimension = 384),
    CONSTRAINT uq_chunk_model UNIQUE (chunk_id, model_id)
);

COMMENT ON TABLE chunk_embeddings IS 'Vector embeddings for document chunks using default 384-dimensional text models.';
COMMENT ON COLUMN chunk_embeddings.embedding IS 'pgvector dense embedding column configured for 384 dimensions.';

-- -----------------------------------------------------------------------------
-- 8. Medical Images (Multimodal Studies & DICOM Metadata)
-- -----------------------------------------------------------------------------
CREATE TABLE medical_images (
    image_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    encounter_id UUID NOT NULL REFERENCES encounters(encounter_id) ON DELETE CASCADE,
    document_id UUID REFERENCES clinical_documents(document_id) ON DELETE SET NULL,
    synthetic_image_num VARCHAR(32) NOT NULL UNIQUE,
    modality VARCHAR(16) NOT NULL,
    body_site VARCHAR(64) NOT NULL,
    synthetic_file_path VARCHAR(512) NOT NULL,
    image_format VARCHAR(16) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_img_modality CHECK (modality IN ('CR', 'DX', 'CT', 'MR', 'US')),
    CONSTRAINT chk_img_format CHECK (image_format IN ('dicom', 'png', 'jpeg', 'nifti'))
);

COMMENT ON TABLE medical_images IS 'Synthetic radiological imaging records and imaging metadata.';

-- -----------------------------------------------------------------------------
-- 9. Medical Image Embeddings (Multimodal Vision Vectors)
-- -----------------------------------------------------------------------------
CREATE TABLE image_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES medical_images(image_id) ON DELETE CASCADE,
    model_id VARCHAR(64) NOT NULL REFERENCES embedding_models(model_id) ON DELETE RESTRICT,
    dimension INT NOT NULL,
    embedding vector(512) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_image_embedding_dim CHECK (dimension = 512),
    CONSTRAINT uq_image_model UNIQUE (image_id, model_id)
);

COMMENT ON TABLE image_embeddings IS 'Multimodal vision vectors (512 dimensions) aligned with medical imaging studies.';
