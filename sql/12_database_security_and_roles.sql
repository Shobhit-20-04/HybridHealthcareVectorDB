-- =============================================================================
-- Migration 12: Database Security, Least-Privilege Roles & Row-Level Security
-- Project: Hybrid Relational–Vector Database Architecture for Healthcare
-- Extension: PostgreSQL pgvector (v0.8.6+)
-- =============================================================================
-- Defines:
--   1. Least-Privilege PostgreSQL Roles:
--      - clinical_admin: Database & Schema Administrator
--      - attending_physician: Department-scoped clinical retrieval
--      - radiologist: Radiology reports & imaging assets retrieval
--      - clinical_researcher: De-identified cohort research retrieval (patients table blocked)
--   2. Dedicated Synthetic Test Login Accounts
--   3. Row-Level Security (RLS) on clinical_documents, document_chunks,
--      chunk_embeddings, medical_images, and image_embeddings
--   4. De-identified Research Projections (views without direct identifiers)
-- =============================================================================

-- =============================================================================
-- SECTION 1: Create Application Roles & Test Login Users
-- =============================================================================
DO $$
BEGIN
    -- 1. Create Functional Roles (NOLOGIN)
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

    -- 2. Create Synthetic Test Login Users (LOGIN)
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'user_admin') THEN
        CREATE USER user_admin WITH PASSWORD 'Admin_Secure_Pass_2026!';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'user_physician') THEN
        CREATE USER user_physician WITH PASSWORD 'Physician_Secure_Pass_2026!';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'user_radiologist') THEN
        CREATE USER user_radiologist WITH PASSWORD 'Radiologist_Secure_Pass_2026!';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'user_researcher') THEN
        CREATE USER user_researcher WITH PASSWORD 'Researcher_Secure_Pass_2026!';
    END IF;
END $$;

-- Assign Functional Roles to Login Users
GRANT clinical_admin TO user_admin;
GRANT attending_physician TO user_physician;
GRANT radiologist TO user_radiologist;
GRANT clinical_researcher TO user_researcher;

-- =============================================================================
-- SECTION 2: Schema Usage & Table-Level Grants
-- =============================================================================

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO clinical_admin, attending_physician, radiologist, clinical_researcher;

-- 2.1 Admin Permissions (Full DML/DDL on all tables and sequences)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO clinical_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO clinical_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO clinical_admin;

-- 2.2 Attending Physician Permissions (Read-only clinical access)
GRANT SELECT ON patients, encounters, diagnoses, clinical_documents, 
                document_chunks, chunk_embeddings, medical_images, 
                image_embeddings, embedding_models TO attending_physician;

-- 2.3 Radiologist Permissions (Read-only imaging & note access)
GRANT SELECT ON patients, encounters, clinical_documents, 
                document_chunks, chunk_embeddings, medical_images, 
                image_embeddings, embedding_models TO radiologist;

-- 2.4 Clinical Researcher Permissions (Least Privilege: NO access to patients table)
REVOKE ALL ON patients FROM clinical_researcher;
GRANT SELECT ON encounters, diagnoses, clinical_documents, 
                document_chunks, chunk_embeddings, embedding_models TO clinical_researcher;
REVOKE ALL ON medical_images, image_embeddings FROM clinical_researcher;

-- =============================================================================
-- SECTION 3: Enable Row-Level Security (RLS) on Clinical & Vector Tables
-- =============================================================================
ALTER TABLE clinical_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE medical_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE image_embeddings ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- SECTION 4: Row-Level Security Policies
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 4.1 Administrator Policies (Full Access)
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS admin_all_docs ON clinical_documents;
CREATE POLICY admin_all_docs ON clinical_documents
    FOR ALL TO clinical_admin USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS admin_all_chunks ON document_chunks;
CREATE POLICY admin_all_chunks ON document_chunks
    FOR ALL TO clinical_admin USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS admin_all_embeddings ON chunk_embeddings;
CREATE POLICY admin_all_embeddings ON chunk_embeddings
    FOR ALL TO clinical_admin USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS admin_all_images ON medical_images;
CREATE POLICY admin_all_images ON medical_images
    FOR ALL TO clinical_admin USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS admin_all_image_embeddings ON image_embeddings;
CREATE POLICY admin_all_image_embeddings ON image_embeddings
    FOR ALL TO clinical_admin USING (true) WITH CHECK (true);

-- -----------------------------------------------------------------------------
-- 4.2 Attending Physician Policies (Departmental Scoping via app.clinician_dept)
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS physician_dept_docs ON clinical_documents;
CREATE POLICY physician_dept_docs ON clinical_documents
    FOR SELECT TO attending_physician
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

DROP POLICY IF EXISTS physician_dept_chunks ON document_chunks;
CREATE POLICY physician_dept_chunks ON document_chunks
    FOR SELECT TO attending_physician
    USING (
        EXISTS (
            SELECT 1 FROM clinical_documents cd
            JOIN encounters e ON cd.encounter_id = e.encounter_id
            WHERE cd.document_id = document_chunks.document_id
              AND (
                  e.department = NULLIF(current_setting('app.clinician_dept', true), '')
                  OR current_setting('app.clinician_dept', true) IS NULL
              )
        )
    );

DROP POLICY IF EXISTS physician_dept_embeddings ON chunk_embeddings;
CREATE POLICY physician_dept_embeddings ON chunk_embeddings
    FOR SELECT TO attending_physician
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

DROP POLICY IF EXISTS physician_dept_images ON medical_images;
CREATE POLICY physician_dept_images ON medical_images
    FOR SELECT TO attending_physician
    USING (
        EXISTS (
            SELECT 1 FROM encounters e
            WHERE e.encounter_id = medical_images.encounter_id
              AND (
                  e.department = NULLIF(current_setting('app.clinician_dept', true), '')
                  OR current_setting('app.clinician_dept', true) IS NULL
              )
        )
    );

DROP POLICY IF EXISTS physician_dept_image_embeddings ON image_embeddings;
CREATE POLICY physician_dept_image_embeddings ON image_embeddings
    FOR SELECT TO attending_physician
    USING (
        EXISTS (
            SELECT 1 FROM medical_images mi
            JOIN encounters e ON mi.encounter_id = e.encounter_id
            WHERE mi.image_id = image_embeddings.image_id
              AND (
                  e.department = NULLIF(current_setting('app.clinician_dept', true), '')
                  OR current_setting('app.clinician_dept', true) IS NULL
              )
        )
    );

-- -----------------------------------------------------------------------------
-- 4.3 Radiologist Policies (Radiology Reports & Imaging Studies Only)
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS radiologist_docs ON clinical_documents;
CREATE POLICY radiologist_docs ON clinical_documents
    FOR SELECT TO radiologist
    USING (
        document_type = 'radiology_report'
    );

DROP POLICY IF EXISTS radiologist_chunks ON document_chunks;
CREATE POLICY radiologist_chunks ON document_chunks
    FOR SELECT TO radiologist
    USING (
        EXISTS (
            SELECT 1 FROM clinical_documents cd
            WHERE cd.document_id = document_chunks.document_id
              AND cd.document_type = 'radiology_report'
        )
    );

DROP POLICY IF EXISTS radiologist_chunk_embeddings ON chunk_embeddings;
CREATE POLICY radiologist_chunk_embeddings ON chunk_embeddings
    FOR SELECT TO radiologist
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
    FOR SELECT TO radiologist
    USING (true);

DROP POLICY IF EXISTS radiologist_image_embeddings ON image_embeddings;
CREATE POLICY radiologist_image_embeddings ON image_embeddings
    FOR SELECT TO radiologist
    USING (true);

-- -----------------------------------------------------------------------------
-- 4.4 Clinical Researcher Policies (Final Status Documents Only; Images Denied)
-- -----------------------------------------------------------------------------
DROP POLICY IF EXISTS researcher_docs ON clinical_documents;
CREATE POLICY researcher_docs ON clinical_documents
    FOR SELECT TO clinical_researcher
    USING (
        status = 'final'
    );

DROP POLICY IF EXISTS researcher_chunks ON document_chunks;
CREATE POLICY researcher_chunks ON document_chunks
    FOR SELECT TO clinical_researcher
    USING (
        EXISTS (
            SELECT 1 FROM clinical_documents cd
            WHERE cd.document_id = document_chunks.document_id
              AND cd.status = 'final'
        )
    );

DROP POLICY IF EXISTS researcher_chunk_embeddings ON chunk_embeddings;
CREATE POLICY researcher_chunk_embeddings ON chunk_embeddings
    FOR SELECT TO clinical_researcher
    USING (
        EXISTS (
            SELECT 1 FROM document_chunks dc
            JOIN clinical_documents cd ON dc.document_id = cd.document_id
            WHERE dc.chunk_id = chunk_embeddings.chunk_id
              AND cd.status = 'final'
        )
    );

-- =============================================================================
-- SECTION 5: De-Identified Views for Safe Research Retrieval
-- =============================================================================

DROP VIEW IF EXISTS deidentified_chunk_embeddings CASCADE;
DROP VIEW IF EXISTS deidentified_document_chunks CASCADE;
DROP VIEW IF EXISTS deidentified_clinical_documents CASCADE;

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
    EXTRACT(YEAR FROM p.date_of_birth)::int AS birth_year,
    EXTRACT(YEAR FROM AGE(e.admission_date, p.date_of_birth))::int AS patient_age
FROM clinical_documents cd
JOIN encounters e ON cd.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id;

GRANT SELECT ON deidentified_clinical_documents TO clinical_researcher;

CREATE OR REPLACE VIEW deidentified_document_chunks AS
SELECT 
    dc.chunk_id,
    dc.document_id,
    dc.chunk_index,
    dc.chunk_text,
    dc.char_length,
    dc.token_count,
    cd.document_type,
    cd.title,
    cd.department,
    cd.gender,
    cd.birth_year,
    cd.patient_age
FROM document_chunks dc
JOIN deidentified_clinical_documents cd ON dc.document_id = cd.document_id;

GRANT SELECT ON deidentified_document_chunks TO clinical_researcher;

CREATE OR REPLACE VIEW deidentified_chunk_embeddings AS
SELECT 
    ce.embedding_id,
    ce.chunk_id,
    ce.model_id,
    ce.dimension,
    ce.embedding,
    dc.chunk_text,
    dc.document_type,
    dc.department,
    dc.patient_age
FROM chunk_embeddings ce
JOIN deidentified_document_chunks dc ON ce.chunk_id = dc.chunk_id;

GRANT SELECT ON deidentified_chunk_embeddings TO clinical_researcher;
