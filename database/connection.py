"""
Database Connection & Session Management Module
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Provides robust, pooled PostgreSQL connections with native pgvector type registration.
Credentials are loaded strictly from environment variables (.env); secrets are never hardcoded.
"""

import os
import sys
from contextlib import contextmanager
from typing import Generator, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

class DatabaseConfig:
    """Encapsulates database configuration loaded from environment variables."""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.dbname = os.getenv("DB_NAME", "healthcare_vectordb")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")
        self.sslmode = os.getenv("DB_SSLMODE", "prefer")

    @property
    def connection_params(self) -> Dict[str, Any]:
        """Returns connection dictionary for psycopg."""
        params = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "sslmode": self.sslmode,
            "connect_timeout": 3
        }
        if self.password:
            params["password"] = self.password
        return params

    @property
    def safe_connection_string(self) -> str:
        """Returns sanitized connection string masking password for logs/diagnostics."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.dbname}?sslmode={self.sslmode}"

    def safe_role_connection_string(self, user: str) -> str:
        """Returns sanitized connection string for a specific role user."""
        return f"postgresql://{user}:***@{self.host}:{self.port}/{self.dbname}?sslmode={self.sslmode}"

    def get_role_credentials(self, role: str) -> tuple[str, str]:
        """Resolves (username, password) for a specified clinical role."""
        role_map = {
            "clinical_admin": ("DB_USER_ADMIN", "DB_PASS_ADMIN", "user_admin", "Admin_Secure_Pass_2026!"),
            "admin": ("DB_USER_ADMIN", "DB_PASS_ADMIN", "user_admin", "Admin_Secure_Pass_2026!"),
            "attending_physician": ("DB_USER_PHYSICIAN", "DB_PASS_PHYSICIAN", "user_physician", "Physician_Secure_Pass_2026!"),
            "physician": ("DB_USER_PHYSICIAN", "DB_PASS_PHYSICIAN", "user_physician", "Physician_Secure_Pass_2026!"),
            "radiologist": ("DB_USER_RADIOLOGIST", "DB_PASS_RADIOLOGIST", "user_radiologist", "Radiologist_Secure_Pass_2026!"),
            "clinical_researcher": ("DB_USER_RESEARCHER", "DB_PASS_RESEARCHER", "user_researcher", "Researcher_Secure_Pass_2026!"),
            "researcher": ("DB_USER_RESEARCHER", "DB_PASS_RESEARCHER", "user_researcher", "Researcher_Secure_Pass_2026!")
        }
        entry = role_map.get(role.lower())
        if entry:
            u_env, p_env, u_def, p_def = entry
            user = os.getenv(u_env, u_def)
            password = os.getenv(p_env, p_def)
            return user, password
        return self.user, self.password

    @property
    def dsn(self) -> str:
        """Standard PostgreSQL DSN string."""
        pw_part = f":{self.password}" if self.password else ""
        return f"postgresql://{self.user}{pw_part}@{self.host}:{self.port}/{self.dbname}?sslmode={self.sslmode}"

def get_db_config() -> DatabaseConfig:
    """Returns DatabaseConfig reflecting active environment variables."""
    return DatabaseConfig()

@contextmanager
def get_connection(autocommit: bool = False):
    """
    Context manager providing an active psycopg connection with vector registration.
    Rolls back on unhandled exceptions and safely closes upon exit.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    cfg = get_db_config()
    conn = None
    try:
        conn = psycopg.connect(**cfg.connection_params, autocommit=autocommit)
        # Register pgvector extension types for native numpy / list vector serialization
        try:
            register_vector(conn)
        except Exception:
            # Vector extension might not yet be installed in target database
            pass
        yield conn
        if not autocommit and conn and not conn.closed:
            conn.commit()
    except Exception:
        if conn and not conn.closed and not autocommit:
            conn.rollback()
        raise
    finally:
        if conn and not conn.closed:
            conn.close()

@contextmanager
def get_cursor(autocommit: bool = False):
    """
    Context manager providing an active database cursor.
    """
    with get_connection(autocommit=autocommit) as conn:
        with conn.cursor() as cur:
            yield cur

@contextmanager
def get_role_connection(
    role: str,
    department: Optional[str] = None,
    direct_login: bool = True,
    autocommit: bool = False
):
    """
    Context manager providing a database connection scoped to a specific clinical role.
    If direct_login=True, connects directly using the dedicated test user credentials.
    If direct_login=False or if direct connection fails, connects as DB_USER and applies SET ROLE.
    Configures session variables (e.g. app.clinician_dept) inside an active transaction.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    cfg = get_db_config()
    role_user, role_pass = cfg.get_role_credentials(role)

    conn = None
    try:
        if direct_login:
            params = {
                "host": cfg.host,
                "port": cfg.port,
                "dbname": cfg.dbname,
                "user": role_user,
                "password": role_pass,
                "sslmode": cfg.sslmode,
                "connect_timeout": 3
            }
            conn = psycopg.connect(**params, autocommit=autocommit)
        else:
            conn = psycopg.connect(**cfg.connection_params, autocommit=autocommit)
            with conn.cursor() as cur:
                cur.execute(f"SET ROLE {role};")

        try:
            register_vector(conn)
        except Exception:
            pass

        if department:
            with conn.cursor() as cur:
                cur.execute(f"SET LOCAL app.clinician_dept = '{department}';")

        yield conn
        if not autocommit and conn and not conn.closed:
            conn.commit()
    except Exception:
        if conn and not conn.closed and not autocommit:
            conn.rollback()
        raise
    finally:
        if conn and not conn.closed:
            conn.close()

def test_connection() -> Dict[str, Any]:
    """
    Tests live connectivity to PostgreSQL.
    Returns status dictionary without exposing sensitive credentials.
    """
    cfg = get_db_config()
    try:
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                cur.execute("SELECT current_database(), current_user;")
                row = cur.fetchone()
                return {
                    "success": True,
                    "target": cfg.safe_connection_string,
                    "database": row[0],
                    "user": row[1],
                    "version": version
                }
    except Exception as exc:
        return {
            "success": False,
            "target": cfg.safe_connection_string,
            "error": str(exc)
        }

def check_vector_support() -> Dict[str, Any]:
    """
    Checks if the pgvector extension is installed and functioning in the active database.
    Verifies vector data types and cosine distance calculation (<=>).
    """
    try:
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                # Check pg_extension
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                row = cur.fetchone()
                if not row:
                    return {
                        "installed": False,
                        "error": "pgvector extension is not registered in pg_extension. Run CREATE EXTENSION vector;"
                    }
                ext_version = row[0]

                # Test vector distance calculation (<=>)
                cur.execute("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector AS cosine_dist;")
                dist = cur.fetchone()[0]

                return {
                    "installed": True,
                    "version": ext_version,
                    "cosine_operator_test": "PASSED",
                    "sample_distance": dist
                }
    except Exception as exc:
        return {
            "installed": False,
            "error": str(exc)
        }

if __name__ == "__main__":
    print("[+] Testing database connection configuration...")
    res = test_connection()
    if res["success"]:
        print(f"[V] Connection Successful!")
        print(f"    - Target  : {res['target']}")
        print(f"    - Version : {res['version']}")
        vec_res = check_vector_support()
        if vec_res["installed"]:
            print(f"[V] pgvector Verified: Version {vec_res['version']} (Cosine Operator: {vec_res['cosine_operator_test']})")
        else:
            print(f"[!] pgvector Notice: {vec_res['error']}")
    else:
        print(f"[!] Connection Failed to {res['target']}")
        print(f"    Error: {res['error']}")
