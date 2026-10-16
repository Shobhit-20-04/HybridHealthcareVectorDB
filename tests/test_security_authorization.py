"""
Unit & Integration Tests for Database-Level Security, RBAC & Row-Level Security (RLS)
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Verifies:
1. Positive authorization: clinical_admin full visibility across all tables and embeddings
2. Positive authorization: attending_physician department-scoped access (Cardiology)
3. Negative authorization: attending_physician cross-department isolation (Neurology blocked)
4. Vector Search Protection: Vector similarity ranking and snippets cannot leak unauthorized departments
5. Positive authorization: radiologist access to radiology reports and medical imaging assets
6. Negative authorization: radiologist blocked from non-imaging clinical documents
7. Positive authorization: clinical_researcher access to de-identified research views
8. Negative authorization: clinical_researcher direct access to patients table raises InsufficientPrivilege
9. Negative authorization: clinical_researcher direct access to medical_images raises InsufficientPrivilege
10. Vector Search Protection: clinical_researcher vector query cannot join patients table to obtain MRN
11. Dedicated Login User Authentication: user_admin, user_physician, user_radiologist, user_researcher
"""

import pytest
import psycopg
from database.connection import get_role_connection, get_db_config
from src.embeddings.text_embedder import TextEmbedder
from src.security.rbac import SecurityContext, ClinicalRole

@pytest.fixture(scope="module")
def text_embedder():
    return TextEmbedder(use_fallback=True)

# =============================================================================
# 1. Administrator Permissions Tests
# =============================================================================

def test_admin_full_visibility():
    """clinical_admin has full visibility across clinical documents, chunks, and embeddings."""
    with get_role_connection("clinical_admin", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clinical_documents;")
            docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
            embeds = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM medical_images;")
            images = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM image_embeddings;")
            img_embeds = cur.fetchone()[0]

    assert docs == 586
    assert chunks == 804
    assert embeds == 804
    assert images == 227
    assert img_embeds == 227

# =============================================================================
# 2. Attending Physician Permissions & RLS Tests
# =============================================================================

def test_physician_departmental_scoping_positive():
    """attending_physician scoped to Cardiology can retrieve Cardiology documents and chunks."""
    with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clinical_documents;")
            docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            chunks = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
            embeds = cur.fetchone()[0]

    assert docs == 64
    assert chunks == 85
    assert embeds == 85

def test_physician_cross_department_isolation_negative():
    """attending_physician scoped to Cardiology CANNOT retrieve Neurology documents."""
    with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cd.document_id 
                FROM clinical_documents cd
                JOIN encounters e ON cd.encounter_id = e.encounter_id
                WHERE e.department = 'Neurology';
            """)
            rows = cur.fetchall()

    assert len(rows) == 0, "Security Violation: Cardiology physician was able to view Neurology documents!"

def test_physician_vector_search_department_isolation(text_embedder):
    """
    Vector similarity search executed under attending_physician (Cardiology)
    MUST NOT return chunks or snippets from other departments, even when the query
    is semantically aligned with an unauthorized department (e.g. Neurology stroke).
    """
    neuro_vec = text_embedder.embed_text("acute ischemic stroke and middle cerebral artery infarct")

    with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    e.department,
                    cd.title,
                    dc.chunk_text,
                    (ce.embedding <=> %s::vector) AS dist
                FROM chunk_embeddings ce
                JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                JOIN clinical_documents cd ON dc.document_id = cd.document_id
                JOIN encounters e ON cd.encounter_id = e.encounter_id
                WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
                ORDER BY ce.embedding <=> %s::vector ASC
                LIMIT 5;
            """, (neuro_vec, neuro_vec))
            results = cur.fetchall()

    assert len(results) > 0, "Expected search results within authorized department"
    for r in results:
        dept, title, snippet, dist = r
        assert dept == "Cardiology", f"Vector Search Leakage: Found result from department {dept}!"
        assert "Neurology" not in title

# =============================================================================
# 3. Radiologist Permissions & RLS Tests
# =============================================================================

def test_radiologist_imaging_and_reports_access_positive():
    """radiologist can view radiology reports, medical images, and image embeddings."""
    with get_role_connection("radiologist", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clinical_documents;")
            rad_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM medical_images;")
            rad_images = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM image_embeddings;")
            rad_img_embeds = cur.fetchone()[0]

    assert rad_docs == 109
    assert rad_images == 227
    assert rad_img_embeds == 227

def test_radiologist_non_imaging_documents_blocked_negative():
    """radiologist CANNOT view discharge summaries or progress notes."""
    with get_role_connection("radiologist", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'discharge_summary';")
            discharge_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'progress_note';")
            progress_count = cur.fetchone()[0]

    assert discharge_count == 0, "Security Violation: Radiologist accessed discharge summaries!"
    assert progress_count == 0, "Security Violation: Radiologist accessed progress notes!"

def test_radiologist_vector_search_isolation(text_embedder):
    """Vector search as radiologist returns strictly radiology report chunks."""
    query_vec = text_embedder.embed_text("cardiac arrest resuscitation emergency progress note")

    with get_role_connection("radiologist", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    cd.document_type,
                    cd.title,
                    (ce.embedding <=> %s::vector) AS dist
                FROM chunk_embeddings ce
                JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                JOIN clinical_documents cd ON dc.document_id = cd.document_id
                WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
                ORDER BY ce.embedding <=> %s::vector ASC
                LIMIT 5;
            """, (query_vec, query_vec))
            results = cur.fetchall()

    assert len(results) > 0
    for r in results:
        doc_type, title, dist = r
        assert doc_type == "radiology_report", f"Vector Search Leakage: Radiologist retrieved {doc_type}!"

# =============================================================================
# 4. Clinical Researcher De-Identification & Access Denial Tests
# =============================================================================

def test_researcher_deidentified_views_access_positive():
    """clinical_researcher can query deidentified_clinical_documents and deidentified_chunk_embeddings."""
    with get_role_connection("clinical_researcher", direct_login=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM deidentified_clinical_documents;")
            deid_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM deidentified_chunk_embeddings;")
            deid_embeds = cur.fetchone()[0]

    assert deid_docs == 586
    assert deid_embeds == 804

def test_researcher_direct_patients_access_denied_negative():
    """clinical_researcher direct access to patients table is rejected with InsufficientPrivilege."""
    with get_role_connection("clinical_researcher", direct_login=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("SELECT * FROM patients LIMIT 1;")

def test_researcher_direct_medical_images_denied_negative():
    """clinical_researcher direct access to medical_images is rejected with InsufficientPrivilege."""
    with get_role_connection("clinical_researcher", direct_login=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("SELECT * FROM medical_images LIMIT 1;")

def test_researcher_vector_join_patients_denied_negative(text_embedder):
    """
    Attempting to perform a vector search that joins patients to discover patient MRN
    under clinical_researcher role is rejected by PostgreSQL table permissions.
    """
    query_vec = text_embedder.embed_text("clinical research study cohort")

    with get_role_connection("clinical_researcher", direct_login=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("""
                    SELECT 
                        p.synthetic_mrn,
                        cd.title,
                        (ce.embedding <=> %s::vector) AS dist
                    FROM chunk_embeddings ce
                    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    JOIN clinical_documents cd ON dc.document_id = cd.document_id
                    JOIN patients p ON cd.patient_id = p.patient_id
                    ORDER BY ce.embedding <=> %s::vector ASC
                    LIMIT 5;
                """, (query_vec, query_vec))

# =============================================================================
# 5. Dedicated User Authentication Tests
# =============================================================================

def test_dedicated_user_connections_and_identity():
    """Verifies each dedicated test user can authenticate and reflects the proper role."""
    cfg = get_db_config()
    roles_to_test = [
        ("user_admin", "Admin_Secure_Pass_2026!"),
        ("user_physician", "Physician_Secure_Pass_2026!"),
        ("user_radiologist", "Radiologist_Secure_Pass_2026!"),
        ("user_researcher", "Researcher_Secure_Pass_2026!")
    ]

    for username, password in roles_to_test:
        params = {
            "host": cfg.host,
            "port": cfg.port,
            "dbname": cfg.dbname,
            "user": username,
            "password": password,
            "sslmode": cfg.sslmode,
            "connect_timeout": 3
        }
        with psycopg.connect(**params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT session_user, current_user;")
                sess_user, cur_user = cur.fetchone()
                assert sess_user == username
                assert cur_user == username

# =============================================================================
# 6. SecurityContext Execution Helper Tests
# =============================================================================

def test_security_context_execution_helper(text_embedder):
    """Verifies SecurityContext.execute_vector_retrieval() returns scoped results."""
    ctx = SecurityContext(user_id="doc-cardio", role=ClinicalRole.PHYSICIAN, department="Cardiology")
    query_vec = text_embedder.embed_text("cardiac telemetry arrhythmia")

    with get_role_connection("attending_physician", direct_login=True) as conn:
        results = ctx.execute_vector_retrieval(conn, query_vec, top_k=3)

    assert len(results) > 0
    for r in results:
        assert r["department"] == "Cardiology"
        assert "snippet" in r
        assert "cosine_distance" in r
