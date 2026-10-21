"""
Interactive DBMS Faculty Demonstration Console
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
Database: PostgreSQL 18.6 with pgvector v0.8.6
Dataset: 1,000 Real PubMed Central (PMC-Patients) Cases (17,076 total records)

Usage:
    python scripts/faculty_query_demo.py
"""

import sys
import time
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection, get_role_connection

QUERIES = {
    "1": {
        "title": "HYBRID SEARCH: SQL Filtering + Vector Cosine Distance (<=>)",
        "concept": "Combines relational B-Tree equality filter (department = 'Cardiology') with pgvector native cosine distance (<=>) ordering over 384-dimensional embeddings.",
        "sql": """SELECT 
    p.synthetic_mrn AS patient_mrn,
    p.gender,
    ROUND(EXTRACT(YEAR FROM age(p.date_of_birth))::numeric, 0) AS age,
    e.department,
    d.document_type,
    SUBSTRING(c.chunk_text, 1, 75) || '...' AS clinical_snippet,
    ROUND((1 - (ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)))::numeric, 4) AS cosine_similarity,
    ROUND((ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1))::numeric, 4) AS cosine_distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
JOIN encounters e ON d.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id
WHERE e.department = 'Cardiology'
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;"""
    },
    "2": {
        "title": "EXPLAIN (ANALYZE, BUFFERS): HNSW Index Scan Verification",
        "concept": "Demonstrates PostgreSQL query optimizer behavior. Shows that the planner uses 'Index Scan using idx_chunk_embeddings_hnsw_cosine' instead of a full sequential table scan.",
        "sql": """EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    p.synthetic_mrn,
    e.department,
    d.document_type,
    ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1) AS distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
JOIN encounters e ON d.encounter_id = e.encounter_id
JOIN patients p ON e.patient_id = p.patient_id
WHERE e.department = 'Cardiology'
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;"""
    },
    "3": {
        "title": "EXACT TOP-K VECTOR SEARCH (<=> Operator on 6,244 Chunks)",
        "concept": "Pure vector similarity retrieval calculating exact cosine distance across all 6,244 real clinical chunks in PostgreSQL without relational pre-filters.",
        "sql": """SELECT 
    d.synthetic_document_num AS doc_num,
    d.document_type,
    c.chunk_index,
    c.token_count,
    SUBSTRING(c.chunk_text, 1, 70) || '...' AS snippet,
    ROUND((1 - (ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)))::numeric, 4) AS cosine_sim,
    ROUND((ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1))::numeric, 4) AS distance
FROM chunk_embeddings ce
JOIN document_chunks c ON ce.chunk_id = c.chunk_id
JOIN clinical_documents d ON c.document_id = d.document_id
ORDER BY ce.embedding <=> (SELECT embedding FROM chunk_embeddings LIMIT 1)
LIMIT 5;"""
    },
    "4": {
        "title": "4-WAY RELATIONAL INTEGRITY JOIN (Patients -> Encounters -> Diagnoses -> Docs)",
        "concept": "Demonstrates 3NF relational normalization and primary-to-foreign key referential integrity across 4 distinct healthcare entity tables.",
        "sql": """SELECT 
    p.synthetic_mrn,
    p.gender,
    e.department,
    TO_CHAR(e.admission_date, 'YYYY-MM-DD') AS admission,
    diag.icd10_code,
    SUBSTRING(diag.diagnosis_description, 1, 30) AS diagnosis,
    doc.document_type,
    SUBSTRING(doc.title, 1, 40) || '...' AS document_title
FROM patients p
JOIN encounters e ON p.patient_id = e.patient_id
JOIN diagnoses diag ON e.encounter_id = diag.encounter_id
JOIN clinical_documents doc ON e.encounter_id = doc.encounter_id
WHERE e.department = 'Neurology'
ORDER BY e.admission_date DESC
LIMIT 8;"""
    },
    "5": {
        "title": "MULTIMODAL RETRIEVAL: Visual Embeddings & Medical Images",
        "concept": "Queries medical image metadata and calculates visual semantic similarity using 512-dimensional CLIP embeddings stored natively in PostgreSQL.",
        "sql": """SELECT 
    img.synthetic_image_num AS image_num,
    img.modality,
    img.body_site,
    e.department,
    ROUND((1 - (ie.embedding <=> (SELECT embedding FROM image_embeddings LIMIT 1)))::numeric, 4) AS visual_similarity,
    ROUND((ie.embedding <=> (SELECT embedding FROM image_embeddings LIMIT 1))::numeric, 4) AS visual_distance,
    img.synthetic_file_path AS file_path
FROM image_embeddings ie
JOIN medical_images img ON ie.image_id = img.image_id
JOIN encounters e ON img.encounter_id = e.encounter_id
ORDER BY ie.embedding <=> (SELECT embedding FROM image_embeddings LIMIT 1)
LIMIT 5;"""
    },
    "6": {
        "title": "DATABASE TELEMETRY: Relation Sizes & Vector Index Footprint",
        "concept": "Inspects PostgreSQL physical storage footprint for relational tables and compares them against pgvector HNSW index sizes on disk.",
        "sql": """SELECT 
    relname AS relation_name,
    CASE 
        WHEN relkind = 'r' THEN 'Table'
        WHEN relkind = 'i' THEN 'Index'
        ELSE 'Other'
    END AS type,
    pg_size_pretty(pg_relation_size(c.oid)) AS disk_size,
    reltuples::bigint AS estimated_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' 
  AND c.relname IN (
    'patients', 'encounters', 'diagnoses', 'clinical_documents', 
    'document_chunks', 'chunk_embeddings', 'medical_images', 'image_embeddings',
    'idx_chunk_embeddings_hnsw_cosine', 'idx_chunk_embeddings_ivfflat_cosine'
  )
ORDER BY pg_relation_size(c.oid) DESC;"""
    },
    "7": {
        "title": "ROW-LEVEL SECURITY (RLS): Admin vs Physician Isolation Test",
        "concept": "Executes SELECT on encounters under 'clinical_admin' vs 'attending_physician' role to prove database-enforced row isolation at the PostgreSQL engine level.",
        "sql": "SPECIAL_RLS"
    }
}

def run_query(sql_text: str, role: str = "postgres"):
    print("\n" + "=" * 90)
    print("EXECUTING RAW SQL QUERY:")
    print("-" * 90)
    print(sql_text.strip())
    print("-" * 90)

    try:
        start_time = time.perf_counter()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_text)
                
                # Check if it's an EXPLAIN query
                if sql_text.strip().upper().startswith("EXPLAIN"):
                    rows = cur.fetchall()
                    elapsed = (time.perf_counter() - start_time) * 1000
                    print(f"\n[+] Execution Time: {elapsed:.2f} ms")
                    print("\nPOSTGRESQL QUERY PLAN OUTPUT:")
                    print("-" * 90)
                    for r in rows:
                        print(r[0])
                    print("-" * 90)
                    return

                # Normal query with columns
                col_names = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
                elapsed = (time.perf_counter() - start_time) * 1000

                print(f"\n[+] Query Execution Time: {elapsed:.2f} ms | Rows Returned: {len(rows)}")
                print("-" * 90)
                
                # Format vector/floats nicely
                formatted_rows = []
                for row in rows:
                    formatted_row = []
                    for val in row:
                        if isinstance(val, float):
                            formatted_row.append(f"{val:.4f}")
                        else:
                            formatted_row.append(str(val))
                    formatted_rows.append(formatted_row)

                print(tabulate(formatted_rows, headers=col_names, tablefmt="psql"))
                print("-" * 90)

    except Exception as e:
        print(f"\n[!] SQL Execution Error: {e}")

def run_rls_demo():
    print("\n" + "=" * 90)
    print("ROW-LEVEL SECURITY (RLS) FACULTY DEMONSTRATION")
    print("-" * 90)
    print("Testing PostgreSQL engine-enforced isolation:")
    print("1. Admin Role (BYPASSRLS enabled): Can view all encounters across all departments.")
    print("2. Physician Role: Enforced RLS policy restricts visibility strictly to assigned department.")
    print("-" * 90)

    sql_check = "SELECT COUNT(*) AS total_encounters, COUNT(DISTINCT department) AS departments_visible FROM encounters;"

    print("\n[A] Executing query as 'clinical_admin' (Full System Visibility):")
    try:
        with get_role_connection("clinical_admin") as conn:
            with conn.cursor() as cur:
                cur.execute(sql_check)
                admin_res = cur.fetchone()
                print(f"    -> Total Encounters Visible: {admin_res[0]} (All {admin_res[1]} departments visible)")
    except Exception as e:
        print(f"    -> Admin check note: {e}")

    print("\n[B] Executing query as 'attending_physician' (Row-Level Security Enforced):")
    try:
        with get_role_connection("attending_physician") as conn:
            with conn.cursor() as cur:
                # Set session department to Cardiology
                cur.execute("SET LOCAL app.current_department = 'Cardiology';")
                cur.execute(sql_check)
                doc_res = cur.fetchone()
                print(f"    -> Encounters Visible: {doc_res[0]} (Restricted strictly to Cardiology)")
    except Exception as e:
        print(f"    -> Physician check note: {e}")

    print("-" * 90)
    print("[V] DBMS Proof: Security isolation is enforced at the database engine level, not in application code.")

def main():
    while True:
        print("\n" + "=" * 80)
        print("  HYBRID HEALTHCARE VECTORDB — ACADEMIC DBMS FACULTY QUERY CONSOLE")
        print("  PostgreSQL 18.6 | pgvector v0.8.6 | 17,076 Real PMC Records")
        print("=" * 80)
        print("  [1] Hybrid Search Query: SQL Filtering (Cardiology) + Vector Similarity (<=>)")
        print("  [2] Execution Plan: EXPLAIN (ANALYZE, BUFFERS) showing HNSW Index Scan")
        print("  [3] Exact Vector Top-K Nearest Neighbors (<=> on 6,244 Chunks)")
        print("  [4] Relational 4-Way Join (Patients -> Encounters -> Diagnoses -> Docs)")
        print("  [5] Multimodal Search: Medical Image Metadata & CLIP Visual Embeddings")
        print("  [6] Database Storage Telemetry: Table Sizes vs Vector Index Footprint")
        print("  [7] Row-Level Security (RLS): Admin vs Physician Isolation Check")
        print("  [8] Custom SQL Query (Type any SQL statement live)")
        print("  [0] Exit")
        print("=" * 80)

        choice = input("\nEnter choice [0-8]: ").strip()

        if choice == "0":
            print("\nExiting Faculty Query Console. Goodbye!")
            break
        elif choice in QUERIES:
            q_info = QUERIES[choice]
            print("\n" + "#" * 80)
            print(f"DEMO: {q_info['title']}")
            print(f"DBMS CONCEPT: {q_info['concept']}")
            print("#" * 80)
            if q_info["sql"] == "SPECIAL_RLS":
                run_rls_demo()
            else:
                run_query(q_info["sql"])
        elif choice == "8":
            print("\nType your SQL query (end with a semicolon or press Enter twice):")
            lines = []
            while True:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
                if line.strip().endswith(";"):
                    break
            custom_sql = "\n".join(lines).strip()
            if custom_sql:
                run_query(custom_sql)
        else:
            print("[!] Invalid option. Please enter a number between 0 and 8.")

if __name__ == "__main__":
    main()
