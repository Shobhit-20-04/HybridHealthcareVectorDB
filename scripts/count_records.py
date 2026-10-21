"""
Database Record Count & Storage Diagnostic CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Audits row counts, table sizes, and manifest synchronization across all relational
and vector schema tables in PostgreSQL.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection, get_db_config

SCHEMA_TABLES = [
    "patients",
    "encounters",
    "diagnoses",
    "clinical_documents",
    "document_chunks",
    "embedding_models",
    "chunk_embeddings",
    "medical_images",
    "image_embeddings"
]

MANIFEST_MAPPING = {
    "patients": "patients",
    "encounters": "encounters",
    "diagnoses": "diagnoses",
    "clinical_documents": "clinical_documents",
    "document_chunks": "document_chunks",
    "medical_images": "medical_images"
}

def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("counts_sanitized") or data.get("counts", {})
        except Exception:
            return {}
    return {}

def main():
    parser = argparse.ArgumentParser(description="Audit row counts and disk sizes of database tables")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    default_manifest = "data/real_pmc/audit_manifest.json" if (PROJECT_ROOT / "data/real_pmc/audit_manifest.json").exists() else "data/processed/audit_manifest.json"
    parser.add_argument(
        "--manifest",
        type=str,
        default=default_manifest,
        help="Path to cohort audit manifest for reconciliation"
    )
    args = parser.parse_args()

    manifest_counts = load_manifest(PROJECT_ROOT / args.manifest)

    table_stats = []
    total_rows = 0

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for table in SCHEMA_TABLES:
                    # Verify table exists in current search_path
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);",
                        (table,)
                    )
                    exists = cur.fetchone()[0]

                    if not exists:
                        table_stats.append({
                            "table": table,
                            "exists": False,
                            "rows": 0,
                            "size": "N/A",
                            "manifest_count": manifest_counts.get(MANIFEST_MAPPING.get(table)),
                            "sync_status": "TABLE MISSING"
                        })
                        continue

                    # Query exact count and formatted disk size
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()[0]
                    total_rows += count

                    cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'));")
                    size_pretty = cur.fetchone()[0]

                    manifest_expected = manifest_counts.get(MANIFEST_MAPPING.get(table))
                    if manifest_expected is not None:
                        if count == manifest_expected:
                            status = "SYNCHRONIZED"
                        else:
                            status = f"DIFF ({count - manifest_expected:+d})"
                    else:
                        status = "N/A"

                    table_stats.append({
                        "table": table,
                        "exists": True,
                        "rows": count,
                        "size": size_pretty,
                        "manifest_count": manifest_expected,
                        "sync_status": status
                    })

    except Exception as e:
        print(f"[!] Database Connection Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        output_payload = {
            "total_rows": total_rows,
            "tables": table_stats
        }
        print(json.dumps(output_payload, indent=2))
        return

    cfg = get_db_config()
    print("\n" + "=" * 84)
    print(f"  POSTGRESQL & PGVECTOR TABLE AUDIT: {cfg.dbname} @ {cfg.host}:{cfg.port}")
    print("=" * 84)
    print(f" {'Table Name':<22} | {'DB Rows':>9} | {'Size':>10} | {'Manifest':>9} | {'Reconciliation':<15}")
    print("-" * 84)

    for item in table_stats:
        manifest_str = str(item["manifest_count"]) if item["manifest_count"] is not None else "-"
        print(
            f" {item['table']:<22} | {item['rows']:>9} | {item['size']:>10} | "
            f"{manifest_str:>9} | {item['sync_status']:<15}"
        )

    print("-" * 84)
    print(f" Total Relational & Vector Records: {total_rows}")
    print("=" * 84 + "\n")

if __name__ == "__main__":
    main()
