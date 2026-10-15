"""
Database package for PostgreSQL + pgvector
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

from database.connection import (
    get_connection,
    get_cursor,
    test_connection,
    check_vector_support,
    get_db_config,
    DatabaseConfig
)

__all__ = [
    "get_connection",
    "get_cursor",
    "test_connection",
    "check_vector_support",
    "get_db_config",
    "DatabaseConfig"
]
