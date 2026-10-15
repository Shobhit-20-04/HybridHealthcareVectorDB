-- =============================================================================
-- Migration 06: Healthcare Access Control & Row-Level Security (RLS)
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- =============================================================================

-- =============================================================================
-- 1. Create Application Clinical Roles
-- =============================================================================
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'clinical_admin') THEN
        CREATE ROLE clinical_admin;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'attending_physician') THEN
        CREATE ROLE attending_physician;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'radiologist') THEN
        CREATE ROLE radiologist;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'clinical_researcher') THEN
        CREATE ROLE clinical_researcher;
    END IF;
END $$;

-- Grant baseline table permissions to roles
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO clinical_admin;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO attending_physician;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO radiologist;
GRANT SELECT ON patients, clinical_documents, document_chunks, chunk_embeddings TO clinical_researcher;

-- =============================================================================
-- 2. Enable Row-Level Security (RLS) on Clinical & Embedding Tables
-- =============================================================================
ALTER TABLE clinical_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE medical_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_embeddings ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- 3. Row-Level Security Policies
-- Session context is injected via: SET LOCAL app.current_user_role = '...'
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Policy 3.1: Admin Full Access Policy
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS admin_all_docs ON clinical_documents;
CREATE POLICY admin_all_docs ON clinical_documents
    FOR ALL
    TO clinical_admin
    USING (true);

DROP POLICY IF EXISTS admin_all_embeddings ON chunk_embeddings;
CREATE POLICY admin_all_embeddings ON chunk_embeddings
    FOR ALL
    TO clinical_admin
    USING (true);

-- -----------------------------------------------------------------------------
-- Policy 3.2: Radiologist Access
-- Radiologists can only view radiology notes, medical images, and image embeddings
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS radiologist_docs ON clinical_documents;
CREATE POLICY radiologist_docs ON clinical_documents
    FOR SELECT
    TO radiologist
    USING (
        document_type = 'radiology_report'
    );

DROP POLICY IF EXISTS radiologist_chunks ON document_chunks;
CREATE POLICY radiologist_chunks ON document_chunks
    FOR SELECT
    TO radiologist
    USING (
        EXISTS (
            SELECT 1 FROM clinical_documents cd 
            WHERE cd.document_id = document_chunks.document_id 
              AND cd.document_type = 'radiology_report'
        )
    );

DROP POLICY IF EXISTS radiologist_chunk_embeddings ON chunk_embeddings;
CREATE POLICY radiologist_chunk_embeddings ON chunk_embeddings
    FOR SELECT
    TO radiologist
    USING (
        EXISTS (
            SELECT 1 FROM document_chunks dc
            JOIN clinical_documents cd ON dc.document_id = cd.document_id
            WHERE dc.chunk_id = chunk_embeddings.chunk_id
              AND cd.document_type = 'radiology_report'
        )
    );

DROP POLICY IF EXISTS radiologist_images ON medical_images;
CREATE POLICY radiologist_images ON medical_images
    FOR SELECT
    TO radiologist
    USING (true);

DROP POLICY IF EXISTS radiologist_image_embeddings ON image_embeddings;
CREATE POLICY radiologist_image_embeddings ON image_embeddings
    FOR SELECT
    TO radiologist
    USING (true);

-- -----------------------------------------------------------------------------
-- Policy 3.3: Attending Physician Department Filtering
-- Evaluates session variable: current_setting('app.clinician_dept', true)
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS physician_dept_docs ON clinical_documents;
CREATE POLICY physician_dept_docs ON clinical_documents
    FOR SELECT
    TO attending_physician
    USING (
        EXISTS (
            SELECT 1 FROM encounters e
            WHERE e.encounter_id = clinical_documents.encounter_id
              AND (
                  e.department = NULLIF(current_setting('app.clinician_dept', true), '')
                  OR current_setting('app.clinician_dept', true) IS NULL
              )
        )
    );

DROP POLICY IF EXISTS physician_dept_embeddings ON chunk_embeddings;
CREATE POLICY physician_dept_embeddings ON chunk_embeddings
    FOR SELECT
    TO attending_physician
    USING (
        EXISTS (
            SELECT 1 FROM document_chunks dc
            JOIN clinical_documents cd ON dc.document_id = cd.document_id
            JOIN encounters e ON cd.encounter_id = e.encounter_id
            WHERE dc.chunk_id = chunk_embeddings.chunk_id
              AND (
                  e.department = NULLIF(current_setting('app.clinician_dept', true), '')
                  OR current_setting('app.clinician_dept', true) IS NULL
              )
        )
    );

-- -----------------------------------------------------------------------------
-- Policy 3.4: Researcher Access (De-Identified View)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW deidentified_clinical_documents AS
SELECT 
    cd.document_id,
    cd.document_type,
    cd.title,
    cd.full_text,
    cd.status,
    e.department,
    e.encounter_type,
    p.gender,
    -- Date shifted or truncated for de-identification
    date_trunc('year', p.date_of_birth) AS birth_year
FROM clinical_documents cd
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id;

GRANT SELECT ON deidentified_clinical_documents TO clinical_researcher;
