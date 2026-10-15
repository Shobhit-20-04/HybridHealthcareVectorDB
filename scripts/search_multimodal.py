"""
Multimodal Healthcare Retrieval CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Provides command-line interface for:
1. Image-to-Image visual similarity retrieval
2. Cross-Modal Text-to-Image semantic retrieval
3. Combined relational metadata filtering (modality, body site, department, age)
4. Query execution plan inspection via EXPLAIN (ANALYZE, BUFFERS)
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.multimodal_search import MultimodalSearchEngine, IncompatibleEmbeddingSpaceError
from src.embeddings.image_embedder import ImageEmbedder

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Healthcare Semantic Search CLI (Image-to-Image & Cross-Modal Text-to-Image)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Path to query image file for Image-to-Image similarity search"
    )
    group.add_argument(
        "--text",
        type=str,
        help="Natural language clinical text query for Cross-Modal Text-to-Image retrieval"
    )

    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve (default: 5)")
    parser.add_argument("--modality", type=str, help="Filter by modality (e.g. DX, CT, MR, US)")
    parser.add_argument("--body-site", type=str, help="Filter by anatomical body site (e.g. Chest, Brain, Abdomen)")
    parser.add_argument("--department", type=str, help="Filter by encounter department (e.g. Cardiology, Neurology)")
    parser.add_argument("--encounter-type", type=str, help="Filter by encounter type (inpatient, outpatient, emergency)")
    parser.add_argument("--min-age", type=int, help="Filter by minimum patient age")
    parser.add_argument("--max-age", type=int, help="Filter by maximum patient age")
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="Multimodal model ID")
    parser.add_argument("--dimension", type=int, default=512, help="Multimodal dimension (default: 512)")
    parser.add_argument("--force-fallback", action="store_true", help="Force deterministic offline fallback embedder")
    parser.add_argument("--explain", action="store_true", help="Include PostgreSQL EXPLAIN ANALYZE execution plan")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    embedder = ImageEmbedder(
        model_name=args.model,
        dimension=args.dimension,
        use_fallback=args.force_fallback
    )
    engine = MultimodalSearchEngine(
        model_id=args.model,
        dimension=args.dimension,
        embedder=embedder
    )

    try:
        if args.image:
            img_path = PROJECT_ROOT / args.image if not Path(args.image).is_absolute() else Path(args.image)
            if not img_path.exists():
                print(f"[!] Error: Image file not found at: {img_path}", file=sys.stderr)
                sys.exit(1)

            res = engine.search_image_to_image(
                query_image=img_path,
                top_k=args.top_k,
                modality=args.modality,
                body_site=args.body_site,
                department=args.department,
                encounter_type=args.encounter_type,
                min_age=args.min_age,
                max_age=args.max_age,
                explain=args.explain
            )
        else:
            res = engine.search_text_to_image(
                query_text=args.text,
                top_k=args.top_k,
                modality=args.modality,
                body_site=args.body_site,
                department=args.department,
                encounter_type=args.encounter_type,
                min_age=args.min_age,
                max_age=args.max_age,
                explain=args.explain
            )

    except IncompatibleEmbeddingSpaceError as e:
        print(f"\n[!] INCOMPATIBLE EMBEDDING SPACE ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Retrieval Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    # Formatted CLI Presentation
    print("\n" + "=" * 92)
    search_title = "IMAGE-TO-IMAGE RETRIEVAL" if res["search_type"] == "image_to_image" else "CROSS-MODAL TEXT-TO-IMAGE RETRIEVAL"
    print(f"  {search_title}")
    print("=" * 92)
    if "query_source" in res:
        print(f"[*] Query Asset    : {res['query_source']}")
    else:
        print(f"[*] Query Text     : '{res['query_text']}'")
    print(f"[*] Latent Space   : {res['model_id']} ({res['dimension']}-d cosine metric)")
    print(f"[*] Active Filters : {res['filters_applied']}")
    print(f"[*] Latency        : Embed: {res['embedding_time_ms']} ms | DB Search: {res['database_time_ms']} ms | Total: {res['total_time_ms']} ms")
    print("-" * 92)

    results = res["results"]
    if not results:
        print("  [!] No matching medical imaging records found satisfying the given query and filters.")
    else:
        print(f" {'Rank':<5} | {'Image ID':<18} | {'Mod / Site':<16} | {'Patient (MRN)':<15} | {'Dept':<14} | {'Dist':<7} | {'Sim':<7}")
        print("-" * 92)
        for r in results:
            mod_site = f"{r['modality']} {r['body_site']}"
            pt_desc = f"{r['synthetic_mrn']} ({r['patient_age']}y)"
            print(
                f" #{r['rank']:<4} | {r['synthetic_image_num']:<18} | {mod_site:<16} | "
                f"{pt_desc:<15} | {r['department']:<14} | {r['cosine_distance']:<7.4f} | {r['cosine_similarity']:<7.4f}"
            )
            if r.get("associated_report_title"):
                print(f"        Associated Report : {r['associated_report_title']}")
            print(f"        Physical Asset    : {r['synthetic_file_path']} (Verified: {r['is_verified_asset']})")

    print("=" * 92)

    if args.explain and res.get("query_plan"):
        print("\n" + "=" * 92)
        print("  POSTGRESQL EXECUTION PLAN: EXPLAIN (ANALYZE, BUFFERS)")
        print("=" * 92)
        for line in res["query_plan"]:
            print(f"  {line}")
        print("=" * 92 + "\n")

if __name__ == "__main__":
    main()
