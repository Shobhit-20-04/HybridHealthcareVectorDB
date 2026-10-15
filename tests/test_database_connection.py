import os
import pytest
from database.connection import DatabaseConfig, get_db_config, test_connection as check_db_connection

def test_database_config_defaults():
    cfg = DatabaseConfig()
    assert cfg.host in ("localhost", "127.0.0.1")
    assert cfg.port == 5432
    assert cfg.dbname == "healthcare_vectordb"
    assert cfg.user == "postgres"

def test_safe_connection_string_masks_password():
    cfg = DatabaseConfig()
    cfg.password = "SuperSecret123"
    safe_str = cfg.safe_connection_string
    assert "SuperSecret123" not in safe_str
    assert "***" in safe_str

def test_connection_params_structure():
    cfg = DatabaseConfig()
    params = cfg.connection_params
    assert "host" in params
    assert "port" in params
    assert "dbname" in params
    assert "user" in params
    assert "sslmode" in params

def test_connection_test_graceful_failure():
    # Setting invalid port to test graceful failure without raw tracebacks
    os.environ["DB_PORT"] = "59999"
    res = check_db_connection()
    assert res["success"] is False
    assert "error" in res
    assert "***" in res["target"]
    os.environ["DB_PORT"] = "5432"
