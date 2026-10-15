"""
PostgreSQL & pgvector Database Health-Check Utility
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Performs comprehensive diagnostic checks:
1. Environment configuration (.env presence & format)
2. Network connectivity (PostgreSQL TCP host:port)
3. Database authentication & connection
4. pgvector extension activation & arithmetic test
5. Schema tables, vector columns & row count audits
6. Row-Level Security (RLS) activation status
Reports any missing prerequisites and exact remediation instructions.
"""

import os
import sys
import socket
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_tcp_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Tests raw TCP connection to PostgreSQL server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False

def run_healthcheck():
    print("\n" + "=" * 78)
    print("  POSTGRESQL & PGVECTOR HEALTH-CHECK DIAGNOSTIC REPORT")
    print("=" * 78)

    # 1. Environment File Check
    env_file = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"
    print("\n[1] Environment Configuration Check:")
    if not env_file.exists():
        print("    [!] .env file NOT FOUND.")
        print(f"        Remediation: Copy {env_example.name} to .env and configure your local credentials:")
        print("        PowerShell: Copy-Item .env.example .env")
        print("        Then edit .env and set DB_PASSWORD=your_postgres_password.")
        has_env = False
    else:
        print("    [V] .env file detected at project root.")
        has_env = True

    from database.connection import get_db_config, test_connection, check_vector_support, get_connection
    cfg = get_db_config()

    print(f"    - DB_HOST : {cfg.host}")
    print(f"    - DB_PORT : {cfg.port}")
    print(f"    - DB_NAME : {cfg.dbname}")
    print(f"    - DB_USER : {cfg.user}")
    print(f"    - DB_PASSWORD : {'[CONFIGURED]' if cfg.password else '[NOT SET / EMPTY]'}")

    # 2. Network Connectivity Check
    print("\n[2] Network Connectivity Check:")
    tcp_ok = check_tcp_port(cfg.host, cfg.port)
    if tcp_ok:
        print(f"    [V] PostgreSQL port {cfg.port} on '{cfg.host}' is REACHABLE and accepting connections.")
    else:
        print(f"    [!] Cannot connect to {cfg.host}:{cfg.port}.")
        print("        Remediation: Verify PostgreSQL service is running:")
        print("        PowerShell: Get-Service -Name postgresql*")
        print("        If stopped: Start-Service postgresql-x64-18 (or appropriate version)")
        print("\n[!] Health-check stopped due to unreachable database port.")
        return

    # 3. Database Authentication & Connection Check
    print("\n[3] Database Authentication & Connection Check:")
    conn_result = test_connection()
    if not conn_result["success"]:
        print(f"    [!] Connection Failed to target: {conn_result['target']}")
        print(f"        PostgreSQL Error: {conn_result['error']}")
        print("\n    Troubleshooting Steps:")
        print("    a. Check if database exists. Create via: createdb -U postgres healthcare_vectordb")
        print("    b. Ensure DB_PASSWORD in .env matches your postgres superuser password.")
        print("    c. Check pg_hba.conf in PostgreSQL data directory.")
        print("\n[!] Health-check halted due to authentication/connection failure.")
        return

    print("    [V] Connected successfully!")
    print(f"    - Connected Database : {conn_result['database']}")
    print(f"    - Connected User     : {conn_result['user']}")
    print(f"    - PostgreSQL Version : {conn_result['version'].split(',')[0]}")

    # 4. pgvector Extension Check
    print("\n[4] pgvector Extension Verification:")
    vec_result = check_vector_support()
    if vec_result["installed"]:
        print(f"    [V] pgvector Extension is ACTIVE (Version {vec_result['version']}).")
        print(f"    [V] Cosine operator (<=>) test PASSED (Distance: {vec_result['sample_distance']}).")
    else:
        print(f"    [!] pgvector is NOT active in database '{cfg.dbname}'.")
        print(f"        Detail: {vec_result['error']}")
        print("\n    Remediation:")
        print("    1. If pgvector binaries are not yet installed in PostgreSQL:")
        print("       Refer to docs/database_setup_guide.md Section 2 for Windows DLL installation.")
        print("    2. If binaries are installed, enable the extension:")
        print("       psql -U postgres -d healthcare_vectordb -c 'CREATE EXTENSION vector;'")

    # 5. Schema Tables & RLS Audit
    print("\n[5] Schema Tables & Row Counts Audit:")
    approved_tables = [
        "patients", "encounters", "diagnoses", "clinical_documents",
        "document_chunks", "embedding_models", "chunk_embeddings",
        "medical_images", "image_embeddings"
    ]

    try:
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                # Query table existence, row counts, and RLS flags
                table_stats = []
                for tbl in approved_tables:
                    cur.execute("""
                        SELECT 
                            EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s),
                            COALESCE((SELECT c.relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = %s), false);
                    """, (tbl, tbl))
                    exists, rls_active = cur.fetchone()
                    if exists:
                        cur.execute(f"SELECT COUNT(*) FROM {tbl};")
                        cnt = cur.fetchone()[0]
                        status = "Present"
                    else:
                        cnt = 0
                        status = "MISSING"

                    table_stats.append((tbl, status, cnt, "ENABLED" if rls_active else "Off"))

                print(f"    {'Table Name':<24} | {'Status':<10} | {'Rows':<8} | {'RLS Policy'}")
                print("    " + "-" * 58)
                missing_tables = []
                for tbl, st, cnt, rls in table_stats:
                    print(f"    {tbl:<24} | {st:<10} | {cnt:<8} | {rls}")
                    if st == "MISSING":
                        missing_tables.append(tbl)

                if missing_tables:
                    print(f"\n    [!] {len(missing_tables)} schema tables are missing.")
                    print("        Run migrations: python scripts/init_database.py")
                    print("        Or via psql: psql -U postgres -d healthcare_vectordb -f sql/init_all.sql")
                else:
                    print("\n    [V] All 9 approved schema tables are present in the database.")

    except Exception as e:
        print(f"    [!] Error during table audit: {e}")

    print("\n" + "=" * 78)
    print("  HEALTH-CHECK FINISHED.")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    run_healthcheck()
