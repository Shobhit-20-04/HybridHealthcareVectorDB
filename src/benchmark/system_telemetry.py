"""
Hardware & Database System Telemetry Collector
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Collects standardized hardware, operating system, PostgreSQL database engine,
and pgvector extension metadata to embed into every benchmark results file.
Guarantees full scientific reproducibility without requiring external packages.
"""

import os
import sys
import ctypes
import platform
from typing import Dict, Any
from database.connection import get_connection

def get_hardware_info() -> Dict[str, Any]:
    """Retrieves host hardware specifications using Python standard library."""
    info = {
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical_cores": os.cpu_count() or 1,
        "total_ram_gb": None
    }

    # Retrieve total physical RAM
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                info["total_ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["total_ram_gb"] = round(kb / (1024 ** 2), 2)
                        break
    except Exception:
        info["total_ram_gb"] = None

    return info

def get_database_info(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Retrieves PostgreSQL database engine, pgvector extension, and vector table telemetry."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. PostgreSQL Version
            cur.execute("SELECT version();")
            pg_version_str = cur.fetchone()[0]

            # 2. pgvector Extension Version
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            vec_row = cur.fetchone()
            pgvector_version = vec_row[0] if vec_row else "NOT_INSTALLED"

            # 3. Model Registration Invariants
            cur.execute(
                "SELECT dimension, distance_metric, modality FROM embedding_models WHERE model_id = %s;",
                (model_id,)
            )
            model_row = cur.fetchone()
            if model_row:
                model_dim, distance_metric, modality = model_row
            else:
                model_dim, distance_metric, modality = 384, "cosine", "text"

            # 4. Storage Footprint & Row Counts
            cur.execute("SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = %s;", (model_id,))
            target_vector_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
            total_vector_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM clinical_documents;")
            total_documents = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            total_chunks = cur.fetchone()[0]

            cur.execute("SELECT pg_size_pretty(pg_relation_size('chunk_embeddings')), pg_relation_size('chunk_embeddings');")
            tbl_size_pretty, tbl_size_bytes = cur.fetchone()

            cur.execute("SELECT pg_size_pretty(pg_total_relation_size('chunk_embeddings')), pg_total_relation_size('chunk_embeddings');")
            total_tbl_size_pretty, total_tbl_size_bytes = cur.fetchone()

    return {
        "postgres_version": pg_version_str,
        "pgvector_version": pgvector_version,
        "embedding_model": {
            "model_id": model_id,
            "dimension": model_dim,
            "distance_metric": distance_metric,
            "pgvector_operator": "<=>" if distance_metric == "cosine" else "<->",
            "modality": modality
        },
        "dataset_statistics": {
            "target_vectors_count": target_vector_count,
            "total_chunk_embeddings": total_vector_count,
            "total_clinical_documents": total_documents,
            "total_document_chunks": total_chunks,
            "table_data_size_pretty": tbl_size_pretty,
            "table_data_size_bytes": tbl_size_bytes,
            "table_total_size_pretty": total_tbl_size_pretty,
            "table_total_size_bytes": total_tbl_size_bytes
        }
    }

def get_full_benchmark_telemetry(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Combines hardware and database telemetry into a unified metadata dictionary."""
    return {
        "hardware": get_hardware_info(),
        "database": get_database_info(model_id=model_id)
    }
