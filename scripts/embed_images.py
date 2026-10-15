"""
Multimodal Medical Image Embedding & Vector Storage CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Executes multimodal pipeline:
1. Generates / synchronizes deterministic synthetic imaging phantoms on disk
2. Audits cryptographic SHA-256 digests and file presence in PostgreSQL
3. Validates model registration and 512-d vector invariants in PostgreSQL pgvector
4. Batch-generates multimodal vision embeddings and upserts to image_embeddings
5. Executes an image-to-image and text-to-image semantic retrieval smoke test
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.image_store import (
    ImageAssetManager,
    IncompatibleEmbeddingSpaceError,
    ImageAssetIntegrityError
)
from src.embeddings.image_embedder import ImageEmbedder
from database.connection import get_connection, get_db_config

def run_multimodal_smoke_test(manager: ImageAssetManager) -> bool:
    """
    Executes a quick multimodal retrieval smoke test:
    1. Image-to-image query (using an existing synthetic image file)
    2. Cross-modal text-to-image query (using a clinical query string)
    """
    print("\n" + "=" * 78)
    print("  RUNNING MULTIMODAL RETRIEVAL SMOKE TEST")
    print("=" * 78)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Pick one verified image as probe
                cur.execute("""
                    SELECT mi.image_id, mi.synthetic_image_num, mi.synthetic_file_path, mi.modality, mi.body_site
                    FROM medical_images mi
                    JOIN image_embeddings ie ON mi.image_id = ie.image_id
                    WHERE ie.model_id = %s
                    LIMIT 1;
                """, (manager.model_id,))
                sample = cur.fetchone()

        if not sample:
            print("[!] Warning: No embedded images found to run smoke test.")
            return False

        probe_id, probe_num, probe_path, modality, body_site = sample
        probe_abs = PROJECT_ROOT / probe_path

        print(f"[*] Probe Image        : {probe_num} ({modality} / {body_site})")
        print(f"[*] Image File Path    : {probe_path}")
        print(f"[*] Model ID           : {manager.model_id} (512-d)")

        # 1. Image-to-Image test
        query_vec = manager.embedder.embed_image_path(probe_abs)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        mi.synthetic_image_num,
                        mi.modality,
                        mi.body_site,
                        p.synthetic_mrn,
                        e.department,
                        (ie.embedding <=> %s::vector) AS cosine_distance
                    FROM image_embeddings ie
                    JOIN medical_images mi ON ie.image_id = mi.image_id
                    JOIN encounters e ON mi.encounter_id = e.encounter_id
                    JOIN patients p ON mi.patient_id = p.patient_id
                    WHERE ie.model_id = %s
                    ORDER BY ie.embedding <=> %s::vector ASC
                    LIMIT 2;
                """, (query_vec, manager.model_id, query_vec))
                matches = cur.fetchall()

        print("\n[V] Image-to-Image Retrieval Results:")
        for rank, row in enumerate(matches, 1):
            img_num, mod, site, mrn, dept, dist = row
            sim = 1.0 - dist
            print(f"    Rank #{rank}: {img_num:<14} | {mod} {site:<12} | MRN: {mrn} | Cosine Dist: {dist:.4f} (Sim: {sim:.4f})")

        # 2. Cross-Modal Text-to-Image test
        query_text = "chest radiograph showing cardiomegaly or pulmonary opacity"
        print(f"\n[*] Cross-Modal Query Text : '{query_text}'")
        text_vec = manager.embedder.embed_query_text(query_text)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        mi.synthetic_image_num,
                        mi.modality,
                        mi.body_site,
                        p.synthetic_mrn,
                        e.department,
                        (ie.embedding <=> %s::vector) AS cosine_distance
                    FROM image_embeddings ie
                    JOIN medical_images mi ON ie.image_id = mi.image_id
                    JOIN encounters e ON mi.encounter_id = e.encounter_id
                    JOIN patients p ON mi.patient_id = p.patient_id
                    WHERE ie.model_id = %s
                    ORDER BY ie.embedding <=> %s::vector ASC
                    LIMIT 2;
                """, (text_vec, manager.model_id, text_vec))
                text_matches = cur.fetchall()

        print("[V] Cross-Modal Text-to-Image Results:")
        for rank, row in enumerate(text_matches, 1):
            img_num, mod, site, mrn, dept, dist = row
            sim = 1.0 - dist
            print(f"    Rank #{rank}: {img_num:<14} | {mod} {site:<12} | MRN: {mrn} | Cosine Dist: {dist:.4f} (Sim: {sim:.4f})")

        print("\n[V] Multimodal smoke test PASSED: Image-to-Image and Text-to-Image retrieval verified.")
        return True

    except Exception as e:
        print(f"[!] Multimodal Smoke Test FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Medical Imaging Asset Generation & pgvector Embedding Pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Multimodal model ID (default: openai/clip-vit-base-patch32)"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=512,
        help="Vector embedding dimension (default: 512)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)"
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Force deterministic offline fallback embedder"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run physical asset integrity audit only without embedding generation"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run image-to-image and cross-modal smoke test after embedding"
    )

    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  MULTIMODAL HEALTHCARE ASSET & VECTOR PIPELINE")
    print("=" * 78)
    print(f"[*] Multimodal Model : {args.model}")
    print(f"[*] Target Dimension : {args.dimension}")
    print(f"[*] Batch Size       : {args.batch_size}")

    cfg = get_db_config()
    print(f"[*] Target Database  : {cfg.safe_connection_string}")

    embedder = ImageEmbedder(
        model_name=args.model,
        dimension=args.dimension,
        use_fallback=args.force_fallback
    )
    manager = ImageAssetManager(
        model_id=args.model,
        dimension=args.dimension,
        embedder=embedder
    )

    # Step 1: Synchronize synthetic image files
    print("\n[1] Synchronizing synthetic imaging phantom files on disk...")
    sync_stats = manager.sync_synthetic_image_files()
    print(f"[+] Total Records Checked : {sync_stats['total_records']}")
    print(f"[+] Newly Generated Files : {sync_stats['generated_on_disk']}")
    print(f"[+] Updated Database Rows : {sync_stats['updated_database']}")

    # Step 2: Audit cryptographic file integrity
    print("\n[2] Executing cryptographic SHA-256 asset integrity audit...")
    audit = manager.verify_asset_integrity()
    print(f"[+] Verified Files        : {audit['verified_count']}")
    print(f"[+] Missing Files         : {audit['missing_count']}")
    print(f"[+] Corrupted Files       : {audit['corrupted_count']}")
    if not audit["audit_passed"]:
        print("[!] Warning: Asset integrity audit encountered issues:")
        for issue in audit["issues"][:5]:
            print(f"    - {issue}")
        if args.verify_only:
            sys.exit(1)
    else:
        print("[V] Cryptographic asset audit PASSED: All physical image files intact.")

    if args.verify_only:
        print("\n" + "=" * 78)
        print("  VERIFICATION COMPLETE (--verify-only specified)")
        print("=" * 78 + "\n")
        return

    # Step 3: Embed and store image vectors
    print("\n[3] Generating 512-d multimodal vision embeddings and storing in pgvector...")
    try:
        t_start = time.perf_counter()
        embed_stats = manager.embed_and_store_images(batch_size=args.batch_size)
        duration = time.perf_counter() - t_start
    except IncompatibleEmbeddingSpaceError as e:
        print(f"\n[!] INCOMPATIBLE EMBEDDING SPACE ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Embedding Pipeline Failure: {e}")
        sys.exit(1)

    print("\n" + "-" * 78)
    print("  MULTIMODAL EMBEDDING SUMMARY")
    print("-" * 78)
    print(f"  Images Processed         : {embed_stats['processed_count']:>6}")
    print(f"  Vectors Upserted         : {embed_stats['upserted_count']:>6}")
    print(f"  Execution Time           : {duration:>6.3f} seconds")
    if embed_stats['upserted_count'] > 0:
        print(f"  Throughput               : {embed_stats['upserted_count'] / duration:>6.1f} img/sec")
    print("-" * 78)

    # Step 4: Smoke Test
    if args.smoke_test or embed_stats['upserted_count'] > 0:
        run_multimodal_smoke_test(manager)

    print("\n" + "=" * 78)
    print("  MULTIMODAL PIPELINE EXECUTION COMPLETED")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
