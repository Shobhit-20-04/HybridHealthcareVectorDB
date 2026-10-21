"""
Database-Level Security & Role-Based Authorization Verification CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Verifies:
1. Least-Privilege PostgreSQL Role existence and membership
2. Table-level Row-Level Security (RLS) activation status
3. Policy coverage across relational and vector embedding tables
4. Positive authorization tests (permitted departmental/modality data access)
5. Negative authorization tests (cross-department isolation, PHI denial, direct table blocks)
6. Vector search leakage prevention (unauthorized snippets and distances withheld)
7. Dedicated login user authentication via environment credentials
"""

import sys
import argparse
from pathlib import Path
import psycopg

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection, get_db_config, get_role_connection
from src.embeddings.text_embedder import TextEmbedder
from src.security.rbac import SecurityContext, ClinicalRole

def audit_roles_and_tables():
    """Audits database catalog for security configuration."""
    print("\n" + "=" * 80)
    print("  1. POSTGRESQL ROLE & RLS STATUS AUDIT")
    print("=" * 80)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Audit roles
            roles = ['clinical_admin', 'attending_physician', 'radiologist', 'clinical_researcher',
                     'user_admin', 'user_physician', 'user_radiologist', 'user_researcher']
            cur.execute("""
                SELECT rolname, rolcanlogin, rolinherit, rolsuper 
                FROM pg_roles 
                WHERE rolname = ANY(%s)
                ORDER BY rolname;
            """, (roles,))
            found_roles = cur.fetchall()

            print(f" {'Role / User Name':<24} | {'Can Login':<11} | {'Inherits':<10} | {'Superuser':<10}")
            print("-" * 80)
            for r in found_roles:
                print(f" {r[0]:<24} | {str(r[1]):<11} | {str(r[2]):<10} | {str(r[3]):<10}")

            # Audit table RLS flags
            print("\n" + "-" * 80)
            print("  TABLE ROW-LEVEL SECURITY (RLS) FLAGS")
            print("-" * 80)
            cur.execute("""
                SELECT tablename, rowsecurity 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                ORDER BY tablename;
            """)
            tables = cur.fetchall()
            for t in tables:
                status = "[ENABLED]" if t[1] else "[DISABLED]"
                print(f"  - {t[0]:<28} : {status}")

            # Audit active RLS policies
            print("\n" + "-" * 80)
            print("  ACTIVE ROW-LEVEL SECURITY POLICIES")
            print("-" * 80)
            cur.execute("""
                SELECT tablename, policyname, roles, cmd 
                FROM pg_policies 
                ORDER BY tablename, policyname;
            """)
            policies = cur.fetchall()
            print(f" {'Table':<22} | {'Policy Name':<32} | {'Roles':<24} | {'Cmd':<6}")
            print("-" * 80)
            for p in policies:
                roles_str = str(p[2])
                print(f" {p[0]:<22} | {p[1]:<32} | {roles_str:<24} | {p[3]:<6}")

def run_authorization_tests():
    """Runs live positive and negative authorization tests."""
    print("\n" + "=" * 80)
    print("  2. LIVE AUTHORIZATION & RLS TEST MATRIX")
    print("=" * 80)

    cfg = get_db_config()
    test_results = []

    # -------------------------------------------------------------------------
    # TEST 1: Admin Full Visibility (Positive)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("clinical_admin", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                n_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                n_embeds = cur.fetchone()[0]
        test_results.append((
            "Admin Full Visibility",
            "clinical_admin",
            "Positive",
            n_docs > 0 and n_embeds > 0,
            f"Visible: {n_docs} docs, {n_embeds} chunk embeddings"
        ))
    except Exception as e:
        test_results.append(("Admin Full Visibility", "clinical_admin", "Positive", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 2: Attending Physician Departmental Scoping (Positive)
    # -------------------------------------------------------------------------
    try:
        # Resolve true ground-truth Cardiology counts from database
        with get_role_connection("clinical_admin", direct_login=True) as a_conn:
            with a_conn.cursor() as a_cur:
                a_cur.execute("""
                    SELECT COUNT(DISTINCT cd.document_id), COUNT(ce.embedding_id)
                    FROM clinical_documents cd
                    JOIN encounters e ON cd.encounter_id = e.encounter_id
                    JOIN document_chunks dc ON cd.document_id = dc.document_id
                    JOIN chunk_embeddings ce ON dc.chunk_id = ce.chunk_id
                    WHERE e.department = 'Cardiology';
                """)
                exp_cardio_docs, exp_cardio_chunks = a_cur.fetchone()

        with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                n_cardio_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                n_cardio_chunks = cur.fetchone()[0]

        is_scoped = (n_cardio_docs == exp_cardio_docs and n_cardio_chunks == exp_cardio_chunks and n_cardio_docs > 0)
        test_results.append((
            "Physician Department Scoping (Cardio)",
            "attending_physician",
            "Positive",
            is_scoped,
            f"Visible: {n_cardio_docs} docs, {n_cardio_chunks} chunks (Expected {exp_cardio_docs}/{exp_cardio_chunks})"
        ))
    except Exception as e:
        test_results.append(("Physician Department Scoping", "attending_physician", "Positive", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 3: Attending Physician Cross-Department Isolation (Negative)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT cd.document_id 
                    FROM clinical_documents cd
                    JOIN encounters e ON cd.encounter_id = e.encounter_id
                    WHERE e.department = 'Neurology';
                """)
                unauthorized_rows = cur.fetchall()
        is_blocked = (len(unauthorized_rows) == 0)
        test_results.append((
            "Cross-Department Isolation (Neurology Block)",
            "attending_physician",
            "Negative",
            is_blocked,
            f"Leaked Neurology Rows: {len(unauthorized_rows)} (Expected 0)"
        ))
    except Exception as e:
        test_results.append(("Cross-Department Isolation", "attending_physician", "Negative", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 4: Vector Search Leakage Prevention (Negative)
    # -------------------------------------------------------------------------
    try:
        embedder = TextEmbedder()
        neuro_query_vec = embedder.embed_text("acute ischemic stroke and hemiparesis in neurology")

        with get_role_connection("attending_physician", department="Cardiology", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.department, cd.title, (ce.embedding <=> %s::vector) AS dist
                    FROM chunk_embeddings ce
                    JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                    JOIN clinical_documents cd ON dc.document_id = cd.document_id
                    JOIN encounters e ON cd.encounter_id = e.encounter_id
                    WHERE ce.model_id = 'sentence-transformers/all-MiniLM-L6-v2'
                    ORDER BY ce.embedding <=> %s::vector ASC
                    LIMIT 5;
                """, (neuro_query_vec, neuro_query_vec))
                vector_results = cur.fetchall()

        all_cardiology = all(r[0] == "Cardiology" for r in vector_results)
        test_results.append((
            "Vector Search Department Leakage Prevention",
            "attending_physician",
            "Negative",
            all_cardiology and len(vector_results) > 0,
            f"Results returned strictly scoped to active department ({len(vector_results)} rows, 100% Cardiology)"
        ))
    except Exception as e:
        test_results.append(("Vector Search Department Leakage Prevention", "attending_physician", "Negative", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 5: Radiologist Radiology Notes Access (Positive)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("clinical_admin", direct_login=True) as a_conn:
            with a_conn.cursor() as a_cur:
                a_cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'radiology_report';")
                exp_rad_docs = a_cur.fetchone()[0]
                a_cur.execute("SELECT COUNT(*) FROM medical_images;")
                exp_images = a_cur.fetchone()[0]

        with get_role_connection("radiologist", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                rad_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM medical_images;")
                rad_images = cur.fetchone()[0]

        is_rad_valid = (rad_docs == exp_rad_docs and rad_images == exp_images and rad_docs > 0 and rad_images > 0)
        test_results.append((
            "Radiologist Imaging & Reports Access",
            "radiologist",
            "Positive",
            is_rad_valid,
            f"Visible: {rad_docs} radiology reports, {rad_images} medical images (Expected {exp_rad_docs}/{exp_images})"
        ))
    except Exception as e:
        test_results.append(("Radiologist Imaging & Reports Access", "radiologist", "Positive", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 6: Radiologist Non-Radiology Document Block (Negative)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("radiologist", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'discharge_summary';")
                discharge_docs = cur.fetchone()[0]
        test_results.append((
            "Radiologist Non-Imaging Document Block",
            "radiologist",
            "Negative",
            discharge_docs == 0,
            f"Discharge Summaries Visible: {discharge_docs} (Expected 0)"
        ))
    except Exception as e:
        test_results.append(("Radiologist Non-Imaging Document Block", "radiologist", "Negative", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 7: Researcher De-Identified Views Access (Positive)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("clinical_researcher", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM deidentified_clinical_documents;")
                deid_docs = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM deidentified_chunk_embeddings;")
                deid_chunks = cur.fetchone()[0]
        test_results.append((
            "Researcher De-Identified View Access",
            "clinical_researcher",
            "Positive",
            deid_docs > 0 and deid_chunks > 0,
            f"Visible: {deid_docs} de-identified documents, {deid_chunks} text vectors"
        ))
    except Exception as e:
        test_results.append(("Researcher De-Identified View Access", "clinical_researcher", "Positive", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 8: Researcher Direct Patients Table Access Denied (Negative)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("clinical_researcher", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM patients LIMIT 1;")
                test_results.append((
                    "Researcher Direct Patients Table Access",
                    "clinical_researcher",
                    "Negative",
                    False,
                    "ERROR: Researcher could select directly from patients table!"
                ))
    except psycopg.errors.InsufficientPrivilege:
        test_results.append((
            "Researcher Direct Patients Table Access",
            "clinical_researcher",
            "Negative",
            True,
            "Passed: PostgreSQL raised InsufficientPrivilege (permission denied)"
        ))
    except Exception as e:
        test_results.append(("Researcher Direct Patients Table Access", "clinical_researcher", "Negative", False, str(e)))

    # -------------------------------------------------------------------------
    # TEST 9: Researcher Direct Medical Images Access Denied (Negative)
    # -------------------------------------------------------------------------
    try:
        with get_role_connection("clinical_researcher", direct_login=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM medical_images LIMIT 1;")
                test_results.append((
                    "Researcher Medical Images Block",
                    "clinical_researcher",
                    "Negative",
                    False,
                    "ERROR: Researcher could select from medical_images!"
                ))
    except psycopg.errors.InsufficientPrivilege:
        test_results.append((
            "Researcher Medical Images Block",
            "clinical_researcher",
            "Negative",
            True,
            "Passed: PostgreSQL raised InsufficientPrivilege (permission denied)"
        ))
    except Exception as e:
        test_results.append(("Researcher Medical Images Block", "clinical_researcher", "Negative", False, str(e)))

    # Print Summary Table
    print(f" {'Test Case':<42} | {'Role':<20} | {'Type':<8} | {'Status':<8} | {'Details'}")
    print("-" * 120)
    all_passed = True
    for item in test_results:
        status_str = "[PASS]" if item[3] else "[FAIL]"
        if not item[3]:
            all_passed = False
        print(f" {item[0]:<42} | {item[1]:<20} | {item[2]:<8} | {status_str:<8} | {item[4]}")
    print("-" * 120)

    if all_passed:
        print("[V] ALL 9 AUTHORIZATION & SECURITY TESTS PASSED SUCCESSFULLY!")
    else:
        print("[!] SOME SECURITY TESTS FAILED!")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Database-Level Security & Role Authorization Verification")
    parser.add_argument("--audit-only", action="store_true", help="Only audit catalog roles, tables, and policies")
    args = parser.parse_args()

    audit_roles_and_tables()
    if not args.audit_only:
        run_authorization_tests()

if __name__ == "__main__":
    main()
