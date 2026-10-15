"""
Database Ingestion CLI - Load Processed Healthcare Cohorts
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Loads validated, deduplicated synthetic clinical cohorts from JSON into PostgreSQL
using atomic transactions, parameterized SQL, and configurable conflict handling.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import DatabaseLoader, IngestionLoaderError
from database.connection import get_db_config, get_connection

def main():
    parser = argparse.ArgumentParser(
        description="Load validated synthetic healthcare data into PostgreSQL database"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to directory containing processed JSON entity files (default: data/processed)"
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        choices=["update", "ignore"],
        default="update",
        help="Conflict strategy for existing records: 'update' (default) or 'ignore'"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the transaction and check constraints, then rollback without writing to disk"
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate all existing clinical tables before loading a fresh cohort"
    )

    args = parser.parse_args()
    data_path = PROJECT_ROOT / args.data_dir

    print("\n" + "=" * 78)
    print("  HEALTHCARE DATABASE INGESTION LOADER")
    print("=" * 78)
    print(f"[*] Source Directory : {data_path}")
    print(f"[*] Conflict Policy  : {args.on_duplicate.upper()}")
    print(f"[*] Dry-Run Mode     : {'ENABLED (Simulate & Rollback)' if args.dry_run else 'DISABLED (Persist to DB)'}")

    if not data_path.exists():
        print(f"[!] Error: Source directory '{data_path}' does not exist.")
        print("[!] Run 'python scripts/generate_data.py' first to generate and sanitize cohort data.")
        sys.exit(1)

    # 1. Quick DB Preflight Check
    cfg = get_db_config()
    print(f"[*] Target Database  : {cfg.safe_connection_string}")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
        print("[V] Database connection verified.")
    except Exception as e:
        print(f"[!] Fatal: Could not connect to PostgreSQL database: {e}")
        print("[!] Verify PostgreSQL is running and credentials in .env are correct.")
        sys.exit(1)

    # 2. Execute Transactional Load
    start_time = time.time()
    try:
        loader = DatabaseLoader(on_duplicate=args.on_duplicate)
        if args.truncate:
            print("[*] Truncating clinical tables before loading...")
            loader.truncate_tables()
        counts = loader.load_from_directory(data_path, dry_run=args.dry_run)
        duration = time.time() - start_time

        print("\n" + "-" * 78)
        print("  INGESTION EXECUTION REPORT")
        print("-" * 78)
        total_rows = sum(counts.values())
        for entity, count in counts.items():
            print(f"    - {entity:<24} : {count:>7} records")
        print("-" * 78)
        print(f"  Total Rows Processed   : {total_rows}")
        print(f"  Execution Time         : {duration:.3f} seconds")
        if total_rows > 0:
            print(f"  Throughput             : {total_rows / duration:.1f} rows/sec")
        print(f"  Transaction Status     : {'ROLLED BACK (Dry-Run)' if args.dry_run else 'COMMITTED'}")
        print("=" * 78 + "\n")

    except IngestionLoaderError as e:
        print(f"\n[!] Loader Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Ingestion Failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
