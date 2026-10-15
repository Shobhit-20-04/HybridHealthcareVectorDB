"""
Unit tests for database schema & DDL validation
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
from pathlib import Path
from scripts.validate_sql import validate_ddl_structure

def test_sql_ddl_validation():
    base_dir = Path(__file__).resolve().parent.parent
    sql_path = base_dir / "sql"
    success = validate_ddl_structure(sql_path)
    assert success is True, "SQL DDL structural validation failed."
