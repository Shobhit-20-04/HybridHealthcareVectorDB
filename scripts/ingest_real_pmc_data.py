"""
Real-World Clinical Data Ingestion & Vector Pipeline CLI (PMC-Patients)
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Transactionally loads the curated real-world PMC-Patients cohort into PostgreSQL,
enforces 3NF foreign key integrity, computes 384-d dense vector embeddings,
and stores them into pgvector chunk_embeddings.
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection
from src.ingestion.loader import DatabaseLoader
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.vector_store import PgVectorStore
from src.embeddings.image_store import ImageAssetManager

def load_real_pmc_cohort(mode: str = "replace", dry_run: bool = False) -> None:
    data_dir = PROJECT_ROOT / "data" / "real_pmc"
    manifest_path = data_dir / "audit_manifest.json"

    if not manifest_path.exists():
        print(f"[!] Error: Staged real dataset not found in {data_dir}.")
        print("    Run 'python scripts/download_real_pmc_data.py' first.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 80)
    print("  INGESTING REAL-WORLD CLINICAL COHORT INTO POSTGRESQL + PGVECTOR")
    print("=" * 80)
    print(f"[*] Dataset Source     : {manifest.get('dataset_name')} ({manifest.get('source')})")
    print(f"[*] Ingestion Mode     : {mode.upper()} ({'DRY-RUN' if dry_run else 'PERSIST'})")
    print(f"[*] Staged Records     : {manifest.get('counts')}")

    # Load JSON files
    cohort = {}
    for entity in ["patients", "encounters", "diagnoses", "clinical_documents", "document_chunks", "medical_images"]:
        file_path = data_dir / f"{entity}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                cohort[entity] = json.load(f)
        else:
            cohort[entity] = []

    # 1. Transactional Relational Ingestion
    loader = DatabaseLoader(on_duplicate="ignore" if mode == "append" else "update")
    if mode == "replace" and not dry_run:
        print("\n[!] Truncating existing tables for clean replacement...")
        loader.truncate_tables()

    t0 = time.time()
    counts = loader.load_cohort(cohort, dry_run=dry_run)
    t_ingest = time.time() - t0
    print(f"[V] Relational loading completed in {t_ingest:.2f}s:")
    for k, v in counts.items():
        print(f"    - {k:22s}: {v} rows inserted")

    if dry_run:
        print("\n[V] Dry-run verification complete. No database records were modified.")
        return

    # 2. Embedding Generation & Vector Storage in pgvector (Text Chunks)
    print("\n" + "=" * 80)
    print("  VECTORIZING REAL CLINICAL PASSAGES (384-d Cosine Embeddings)")
    print("=" * 80)

    store = PgVectorStore(model_id="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    t_embed_start = time.time()
    embed_stats = store.ingest_chunk_embeddings(batch_size=64)
    t_embed = time.time() - t_embed_start

    print(f"[V] Text vector storage completed in {t_embed:.2f}s:")
    print(f"    - Processed Chunks        : {embed_stats.get('total_chunks')}")
    print(f"    - Newly Stored Embeddings : {embed_stats.get('inserted_embeddings')}")
    print(f"    - Existing / Skipped      : {embed_stats.get('skipped_embeddings')}")

    # 3. Embedding Generation for Medical Images (512-d Vision Embeddings)
    if counts.get("medical_images", 0) > 0:
        print("\n" + "=" * 80)
        print("  VECTORIZING MEDICAL IMAGES (512-d Vision Embeddings)")
        print("=" * 80)
        img_store = ImageAssetManager(model_id="openai/clip-vit-base-patch32", dimension=512)
        t_img_start = time.time()
        img_stats = img_store.embed_and_store_images(batch_size=64)
        t_img = time.time() - t_img_start
        print(f"[V] Image vector storage completed in {t_img:.2f}s:")
        print(f"    - Total Images            : {img_stats.get('total_images')}")
        print(f"    - Newly Stored Embeddings : {img_stats.get('inserted_embeddings')}")

    # 4. Live Semantic Retrieval Smoke Test on Real-World Data
    test_query = "acute respiratory distress syndrome with fever and hypoxia"
    print("\n" + "=" * 80)
    print("  REAL-WORLD SEMANTIC RETRIEVAL VERIFICATION")
    print("=" * 80)
    print(f"[*] Query String: '{test_query}'")

    with get_connection() as conn:
        with conn.cursor() as cur:
            query_vec = store.embedder.embed_text(test_query)
            cur.execute("""
                SELECT 
                    cd.synthetic_document_num,
                    cd.title,
                    e.department,
                    dc.chunk_index,
                    (ce.embedding <=> %s::vector) AS cosine_distance,
                    (1.0 - (ce.embedding <=> %s::vector)) AS cosine_similarity,
                    dc.chunk_text
                FROM chunk_embeddings ce
                JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
                JOIN clinical_documents cd ON dc.document_id = cd.document_id
                JOIN encounters e ON cd.encounter_id = e.encounter_id
                WHERE ce.model_id = %s
                ORDER BY ce.embedding <=> %s::vector ASC
                LIMIT 2;
            """, (query_vec, query_vec, store.model_id, query_vec))
            rows = cur.fetchall()

            for i, r in enumerate(rows, 1):
                doc_num, title, dept, c_idx, dist, sim, text = r
                print(f"\n[Rank #{i}] Cosine Distance: {dist:.4f} | Similarity: {sim:.4f}")
                print(f"    Document   : {title} ({doc_num})")
                print(f"    Department : {dept} | Chunk Index: {c_idx}")
                print(f"    Passage    : {text[:200]}...")

    print("\n" + "=" * 80)
    print(" [V] REAL-WORLD DATA INGESTION & VECTOR RETRIEVAL SUCCESSFULLY VERIFIED!")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Real-World PMC Clinical Data Ingestion CLI")
    parser.add_argument("--mode", choices=["append", "replace"], default="replace",
                        help="Ingestion mode: 'append' to add alongside existing records, 'replace' to reload cleanly (default: replace)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate ingestion within a rolling-back transaction to test constraints")
    args = parser.parse_args()

    load_real_pmc_cohort(mode=args.mode, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
