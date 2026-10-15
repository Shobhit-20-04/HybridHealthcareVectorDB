"""
Database Initialization & Migration Runner
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes SQL migration scripts against PostgreSQL using environment variables.
Provides diagnostic prerequisite checks for PostgreSQL service and pgvector extension.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "healthcare_vectordb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_SSLMODE = os.getenv("DB_SSLMODE", "prefer")

def check_psycopg_available():
    try:
        import psycopg
        return "psycopg"
    except ImportError:
        try:
            import psycopg2
            return "psycopg2"
        except ImportError:
            return None

def run_migrations():
    driver = check_psycopg_available()
    if not driver:
        print("[!] ERROR: Neither 'psycopg' (v3) nor 'psycopg2' is installed in your Python environment.")
        print("    Install dependencies via: pip install -r requirements.txt")
        sys.exit(1)

    print(f"[+] Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME} (User: {DB_USER}) using {driver}...")
    
    conn_params = {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }

    try:
        if driver == "psycopg":
            import psycopg
            conn = psycopg.connect(**conn_params, autocommit=True)
        else:
            import psycopg2
            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True
    except Exception as exc:
        print(f"\n[!] Database Connection Failed: {exc}")
        print("\nPrerequisite Check Checklist:")
        print(f"1. Is PostgreSQL running on {DB_HOST}:{DB_PORT}?")
        print(f"2. Does database '{DB_NAME}' exist? (Create via: createdb -U {DB_USER} {DB_NAME})")
        print(f"3. Are credentials in .env correct?")
        sys.exit(1)

    cursor = conn.cursor()
    base_dir = Path(__file__).resolve().parent.parent
    sql_dir = base_dir / "sql"

    # Core schema migrations in deterministic dependency order
    core_migrations = [
        "01_extensions.sql",
        "02_schema.sql",
        "03_indexes.sql",
        "04_seed_embedding_models.sql",
        "06_security_rls.sql"
    ]

    print(f"[+] Executing {len(core_migrations)} core schema migrations in order:")
    for migration_name in core_migrations:
        sql_file = sql_dir / migration_name
        if not sql_file.exists():
            print(f"[!] Warning: Migration file {migration_name} not found, skipping.")
            continue

        print(f"    -> Applying {migration_name}...", end=" ", flush=True)
        with open(sql_file, "r", encoding="utf-8") as f:
            sql_content = f.read()

        # Sanitize psql meta-commands (lines starting with backslash like \echo, \i, \gexec)
        clean_lines = [
            line for line in sql_content.splitlines() 
            if not line.strip().startswith("\\")
        ]
        clean_sql = "\n".join(clean_lines)

        try:
            cursor.execute(clean_sql)
            print("DONE")
        except Exception as err:
            print(f"FAILED!\n[!] Error applying {migration_name}: {err}")
            cursor.close()
            conn.close()
            sys.exit(1)

    # Verification query
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n[V] Database initialized successfully! Active public tables ({len(tables)}):")
    for tbl in tables:
        print(f"    - {tbl}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_migrations()
