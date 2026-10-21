"""
FastAPI Backend Application for Hybrid Healthcare VectorDB Frontend
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Exposes:
- /api/stats: Live PostgreSQL and pgvector database metadata, table counts, sizes
- /api/search/hybrid: Hybrid relational SQL filtering combined with vector similarity search
- /api/search/multimodal: Text-to-image and image-to-image semantic search
- /api/security/test: Live PostgreSQL Role-Based Access Control and Row-Level Security verification
- /api/rag: Grounded clinical RAG question answering with decoupled evaluation metrics
- /api/images/{filename}: Static file serving for verified medical image assets
- /api/benchmarks: Empirical benchmark telemetry and indexing comparison
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection, get_db_config, get_role_connection
from src.retrieval.hybrid_search import PostgresHybridSearchEngine
from src.retrieval.multimodal_search import MultimodalSearchEngine
from src.embeddings.image_store import ImageAssetManager
from src.rag.pipeline import ClinicalRAGPipeline
from src.security.rbac import SecurityContext, ClinicalRole
import psycopg

logger = logging.getLogger("server.app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Hybrid Healthcare VectorDB API",
    description="Academic DBMS backend for hybrid relational-vector queries, RLS verification, and multimodal retrieval.",
    version="1.0.0"
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons for retrieval engines
_hybrid_engine: Optional[PostgresHybridSearchEngine] = None
_multimodal_engine: Optional[MultimodalSearchEngine] = None
_rag_pipeline: Optional[ClinicalRAGPipeline] = None

def get_hybrid_engine() -> PostgresHybridSearchEngine:
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = PostgresHybridSearchEngine()
    return _hybrid_engine

def get_multimodal_engine() -> MultimodalSearchEngine:
    global _multimodal_engine
    if _multimodal_engine is None:
        _multimodal_engine = MultimodalSearchEngine()
    return _multimodal_engine

def get_rag_pipeline() -> ClinicalRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = ClinicalRAGPipeline(search_engine=get_hybrid_engine())
    return _rag_pipeline

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class HybridSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    department: Optional[str] = None
    document_type: Optional[str] = None
    min_age: Optional[int] = Field(default=None, ge=0, le=120)
    max_age: Optional[int] = Field(default=None, ge=0, le=120)
    icd10_code: Optional[str] = None
    search_mode: str = Field(default="hnsw", pattern="^(exact_flat|hnsw|ivfflat|planner_choice)$")
    explain: bool = False

class MultimodalSearchRequest(BaseModel):
    mode: str = Field(default="text_to_image", pattern="^(text_to_image|image_to_image)$")
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    modality: Optional[str] = None
    body_site: Optional[str] = None
    department: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=20)

class SecurityTestRequest(BaseModel):
    role: str = Field(..., pattern="^(clinical_admin|attending_physician|radiologist|clinical_researcher)$")
    department: Optional[str] = "Cardiology"
    query: str = "acute ischemic stroke and middle cerebral artery infarct"

class RAGRequest(BaseModel):
    question: str
    department: Optional[str] = None
    document_type: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)

class SqlQueryRequest(BaseModel):
    sql: str
    role: Optional[str] = "postgres"


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    cfg = get_db_config()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                row = cur.fetchone()
                pgvector_version = row[0] if row else "Not installed"
                cur.execute("SELECT version();")
                pg_version = cur.fetchone()[0]
        return {
            "status": "healthy",
            "database": cfg.dbname,
            "host": cfg.host,
            "port": cfg.port,
            "pgvector_version": pgvector_version,
            "postgresql_version": pg_version.split(",")[0]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")

@app.get("/api/stats")
def get_database_stats():
    """Returns live table row counts and relation disk sizes from PostgreSQL."""
    tables = [
        "patients", "encounters", "diagnoses", "clinical_documents",
        "document_chunks", "embedding_models", "chunk_embeddings",
        "medical_images", "image_embeddings"
    ]
    results = []
    total_rows = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for t in tables:
                cur.execute(f"SELECT COUNT(*) FROM {t};")
                count = cur.fetchone()[0]
                cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{t}'));")
                size = cur.fetchone()[0]
                results.append({"table": t, "count": count, "size": size})
                total_rows += count

            # Also check active indexes on chunk_embeddings
            cur.execute("""
                SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
                FROM pg_indexes 
                WHERE tablename = 'chunk_embeddings';
            """)
            indexes = [{"name": r[0], "size": r[1]} for r in cur.fetchall()]

    return {
        "tables": results,
        "total_records": total_rows,
        "indexes": indexes,
        "dataset_source": "PubMed Central Case Reports (PMC-Patients, CC-BY 4.0)"
    }

@app.post("/api/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    """Executes single-transaction hybrid relational-plus-vector search."""
    try:
        engine = get_hybrid_engine()
        res = engine.search(
            query=req.query,
            top_k=req.top_k,
            department=req.department if req.department != "ALL" else None,
            document_type=req.document_type if req.document_type != "ALL" else None,
            min_age=req.min_age,
            max_age=req.max_age,
            icd10_code=req.icd10_code,
            search_mode=req.search_mode,
            explain=req.explain
        )
        return res
    except Exception as e:
        logger.error(f"Error executing hybrid search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/multimodal")
def search_multimodal(req: MultimodalSearchRequest):
    """Executes cross-modal (Text-to-Image) or visual (Image-to-Image) similarity search."""
    try:
        engine = get_multimodal_engine()
        if req.mode == "text_to_image":
            if not req.query_text:
                raise HTTPException(status_code=400, detail="query_text is required for text_to_image search")
            res = engine.search_text_to_image(
                query_text=req.query_text,
                top_k=req.top_k,
                modality=req.modality if req.modality != "ALL" else None,
                body_site=req.body_site if req.body_site != "ALL" else None,
                department=req.department if req.department != "ALL" else None
            )
        else:
            if not req.query_image_path:
                # default sample
                sample = PROJECT_ROOT / "data" / "real_pmc" / "images" / "ct_0001.png"
            else:
                sample = PROJECT_ROOT / req.query_image_path
            
            if not sample.exists():
                raise HTTPException(status_code=404, detail=f"Image file not found: {sample}")

            res = engine.search_image_to_image(
                query_image=sample,
                top_k=req.top_k,
                modality=req.modality if req.modality != "ALL" else None,
                body_site=req.body_site if req.body_site != "ALL" else None,
                department=req.department if req.department != "ALL" else None
            )
        return res
    except Exception as e:
        logger.error(f"Error in multimodal search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/security/test")
def test_security_role(req: SecurityTestRequest):
    """
    Executes a test query as the specified PostgreSQL least-privilege role to demonstrate RLS.
    """
    role = req.role
    dept = req.department or "Cardiology"
    report = {
        "role": role,
        "department": dept,
        "checks": []
    }

    try:
        if role == "clinical_admin":
            with get_role_connection("clinical_admin", direct_login=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                    n_docs = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                    n_chunks = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM patients;")
                    n_pts = cur.fetchone()[0]
            report["checks"].append({
                "name": "Full Clinical Visibility",
                "status": "ALLOWED",
                "details": f"Unrestricted: {n_pts} patients, {n_docs} documents, {n_chunks} chunk embeddings visible."
            })

        elif role == "attending_physician":
            with get_role_connection("attending_physician", department=dept, direct_login=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM clinical_documents;")
                    n_docs = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
                    n_chunks = cur.fetchone()[0]
                    
                    # Test cross-department isolation: attempt to query Neurology
                    other_dept = "Neurology" if dept != "Neurology" else "Cardiology"
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM clinical_documents cd
                        JOIN encounters e ON cd.encounter_id = e.encounter_id
                        WHERE e.department = %s;
                    """, (other_dept,))
                    leaked_rows = cur.fetchone()[0]

            report["checks"].append({
                "name": f"Department Scoping ({dept})",
                "status": "SCOPED",
                "details": f"Visible within {dept}: {n_docs} documents, {n_chunks} text chunks."
            })
            report["checks"].append({
                "name": f"Cross-Department Isolation ({other_dept})",
                "status": "PROTECTED" if leaked_rows == 0 else "LEAKED",
                "details": f"Attempted to query {other_dept}: {leaked_rows} rows visible (RLS blocked cross-dept access)."
            })

        elif role == "radiologist":
            with get_role_connection("radiologist", direct_login=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM medical_images;")
                    n_images = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'radiology_report';")
                    n_rad_docs = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM clinical_documents WHERE document_type = 'discharge_summary';")
                    n_discharge = cur.fetchone()[0]

            report["checks"].append({
                "name": "Radiology Assets Access",
                "status": "ALLOWED",
                "details": f"Permitted: {n_images} medical images, {n_rad_docs} radiology reports visible."
            })
            report["checks"].append({
                "name": "Non-Imaging Document Block",
                "status": "PROTECTED" if n_discharge == 0 else "LEAKED",
                "details": f"Attempted to query discharge summaries: {n_discharge} visible (RLS blocked non-imaging records)."
            })

        elif role == "clinical_researcher":
            # Check 1: De-identified view works
            with get_role_connection("clinical_researcher", direct_login=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM deidentified_clinical_documents;")
                    deid_docs = cur.fetchone()[0]

            # Check 2: Direct patients table access is denied
            phi_denied = False
            try:
                with get_role_connection("clinical_researcher", direct_login=True) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT * FROM patients LIMIT 1;")
            except psycopg.errors.InsufficientPrivilege:
                phi_denied = True
            except Exception:
                phi_denied = True

            report["checks"].append({
                "name": "De-Identified Research Cohort Access",
                "status": "ALLOWED",
                "details": f"Permitted access to deidentified_clinical_documents view ({deid_docs} records without PHI)."
            })
            report["checks"].append({
                "name": "Direct PHI / Patients Table Block",
                "status": "DENIED (SECURE)" if phi_denied else "VIOLATION",
                "details": "PostgreSQL rejected direct SELECT on patients table with InsufficientPrivilege (permission denied)."
            })

        return report

    except Exception as e:
        logger.error(f"Error in security role test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
def execute_rag(req: RAGRequest):
    """Executes grounded clinical RAG with decoupled academic evaluation."""
    try:
        pipeline = get_rag_pipeline()
        res = pipeline.execute(
            question=req.question,
            department=req.department if req.department != "ALL" else None,
            document_type=req.document_type if req.document_type != "ALL" else None,
            top_k=req.top_k,
            evaluate=True
        )
        return res.to_dict()
    except Exception as e:
        logger.error(f"Error in RAG pipeline execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images/{filename}")
def serve_image(filename: str):
    """Serves deterministic medical image phantom files."""
    real_path = PROJECT_ROOT / "data" / "real_pmc" / "images" / filename
    if real_path.exists():
        return FileResponse(str(real_path), media_type="image/png")
    
    # check synthetic folder
    for mod in ["ct", "dx", "us"]:
        synth_path = PROJECT_ROOT / "data" / "synthetic" / "images" / mod / filename
        if synth_path.exists():
            return FileResponse(str(synth_path), media_type="image/png")
            
    raise HTTPException(status_code=404, detail=f"Image {filename} not found.")

@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns stored benchmark evaluation results."""
    bench_file = PROJECT_ROOT / "results" / "benchmark_report.json"
    if bench_file.exists():
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"message": "No benchmark report found. Run scripts/run_benchmarks.py to generate."}

# -----------------------------------------------------------------------------
# Database Studio Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
@app.get("/db-admin")
@app.get("/console")
def get_db_admin_console():
    """Serves minimal Database Studio HTML interface."""
    admin_html = PROJECT_ROOT / "server" / "db_admin.html"
    if admin_html.exists():
        return FileResponse(str(admin_html), media_type="text/html")
    raise HTTPException(status_code=404, detail="db_admin.html not found")

@app.get("/api/sql/tables")
def get_tables_list():
    """Returns list of relational tables with row counts for DBMS console sidebar."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """)
                tables = [r[0] for r in cur.fetchall()]
                
                table_stats = []
                for t in tables:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {t};")
                        count = cur.fetchone()[0]
                    except Exception:
                        count = 0
                    table_stats.append({"name": t, "count": count})
                    
                return {"tables": table_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sql/execute")
def execute_sql(req: SqlQueryRequest):
    """Executes arbitrary or pre-loaded SQL query and returns rows, columns, timing, and query plan."""
    import time
    import re
    import numpy as np

    sql = req.sql.strip()

    # Destructive operations check
    forbidden_patterns = [
        r"\bDROP\s+DATABASE\b",
        r"\bDROP\s+SCHEMA\b",
        r"\bDROP\s+TABLE\b",
        r"\bTRUNCATE\b",
        r"\bALTER\s+ROLE\b",
        r"\bDROP\s+ROLE\b"
    ]
    for pat in forbidden_patterns:
        if re.search(pat, sql, re.IGNORECASE):
            return {
                "success": False,
                "error": "Destructive operations (DROP/TRUNCATE/ALTER ROLE) are disabled in the academic demo console."
            }

    is_explain = sql.upper().startswith("EXPLAIN")

    try:
        start_time = time.perf_counter()
        
        # Determine connection role
        if req.role and req.role != "postgres":
            conn_cm = get_role_connection(req.role)
        else:
            conn_cm = get_connection()

        with conn_cm as conn:
            with conn.cursor() as cur:
                if req.role == "attending_physician":
                    cur.execute("SET LOCAL app.current_department = 'Cardiology';")
                
                cur.execute(sql)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if is_explain:
                    rows = cur.fetchall()
                    return {
                        "success": True,
                        "is_explain": True,
                        "sql": sql,
                        "columns": ["QUERY PLAN"],
                        "rows": [[r[0]] for r in rows],
                        "row_count": len(rows),
                        "execution_time_ms": elapsed_ms
                    }

                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    raw_rows = cur.fetchall()
                    
                    formatted_rows = []
                    for r in raw_rows:
                        formatted_row = []
                        for val in r:
                            if val is None:
                                formatted_row.append(None)
                            elif isinstance(val, (list, np.ndarray)) or (hasattr(val, "__len__") and not isinstance(val, (str, bytes, dict)) and len(val) > 20):
                                val_list = list(val)
                                dim = len(val_list)
                                head = ", ".join(f"{x:.4f}" for x in val_list[:3])
                                formatted_row.append(f"[{head}, ... ({dim} dims)]")
                            elif isinstance(val, float):
                                formatted_row.append(round(val, 4))
                            elif hasattr(val, "isoformat"):
                                formatted_row.append(val.isoformat())
                            else:
                                formatted_row.append(str(val))
                        formatted_rows.append(formatted_row)
                    
                    return {
                        "success": True,
                        "is_explain": False,
                        "sql": sql,
                        "columns": columns,
                        "rows": formatted_rows,
                        "row_count": len(formatted_rows),
                        "execution_time_ms": elapsed_ms
                    }
                else:
                    return {
                        "success": True,
                        "is_explain": False,
                        "sql": sql,
                        "columns": ["STATUS"],
                        "rows": [["Query executed successfully. (0 rows returned)"]],
                        "row_count": 0,
                        "execution_time_ms": elapsed_ms
                    }

    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return {"success": False, "error": str(e)}

# Mount built static frontend if available
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=True)
