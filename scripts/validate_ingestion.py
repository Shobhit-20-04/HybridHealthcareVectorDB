"""
Database Ingestion Integrity & Referential Audit CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Performs comprehensive post-ingestion verification:
1. Referential Integrity (zero orphaned children across all foreign key relationships)
2. Domain Invariants (temporal consistency, chunk length invariants, contiguous indices)
3. Uniqueness and Duplicate Invariants
4. Cryptographic Provenance Metadata verification
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_connection, get_db_config

def run_integrity_audit() -> bool:
    """
    Executes a comprehensive suite of SQL verification checks against the live database.
    Returns True if all checks pass, False if any check fails.
    """
    cfg = get_db_config()
    print("\n" + "=" * 84)
    print(f"  POST-INGESTION INTEGRITY & REFERENTIAL AUDIT")
    print(f"  Database: {cfg.dbname} | Host: {cfg.host}:{cfg.port}")
    print("=" * 84)

    checks = [
        # 1. Foreign Key / Referential Integrity Checks
        {
            "category": "Referential Integrity",
            "name": "Orphan Encounters (missing patient)",
            "sql": "SELECT COUNT(*) FROM encounters e LEFT JOIN patients p ON e.patient_id = p.patient_id WHERE p.patient_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Diagnoses (missing encounter)",
            "sql": "SELECT COUNT(*) FROM diagnoses d LEFT JOIN encounters e ON d.encounter_id = e.encounter_id WHERE e.encounter_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Diagnoses (missing patient)",
            "sql": "SELECT COUNT(*) FROM diagnoses d LEFT JOIN patients p ON d.patient_id = p.patient_id WHERE p.patient_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Documents (missing encounter)",
            "sql": "SELECT COUNT(*) FROM clinical_documents cd LEFT JOIN encounters e ON cd.encounter_id = e.encounter_id WHERE e.encounter_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Documents (missing patient)",
            "sql": "SELECT COUNT(*) FROM clinical_documents cd LEFT JOIN patients p ON cd.patient_id = p.patient_id WHERE p.patient_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Chunks (missing document)",
            "sql": "SELECT COUNT(*) FROM document_chunks dc LEFT JOIN clinical_documents cd ON dc.document_id = cd.document_id WHERE cd.document_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Medical Images (missing patient)",
            "sql": "SELECT COUNT(*) FROM medical_images mi LEFT JOIN patients p ON mi.patient_id = p.patient_id WHERE p.patient_id IS NULL;",
            "expected": 0
        },
        {
            "category": "Referential Integrity",
            "name": "Orphan Medical Images (missing encounter)",
            "sql": "SELECT COUNT(*) FROM medical_images mi LEFT JOIN encounters e ON mi.encounter_id = e.encounter_id WHERE e.encounter_id IS NULL;",
            "expected": 0
        },

        # 2. Duplicate / Uniqueness Invariants
        {
            "category": "Uniqueness Constraints",
            "name": "Duplicate Synthetic MRNs in Patients",
            "sql": "SELECT COUNT(*) - COUNT(DISTINCT synthetic_mrn) FROM patients;",
            "expected": 0
        },
        {
            "category": "Uniqueness Constraints",
            "name": "Duplicate Synthetic Encounter Numbers",
            "sql": "SELECT COUNT(*) - COUNT(DISTINCT synthetic_encounter_num) FROM encounters;",
            "expected": 0
        },
        {
            "category": "Uniqueness Constraints",
            "name": "Duplicate Synthetic Document Numbers",
            "sql": "SELECT COUNT(*) - COUNT(DISTINCT synthetic_document_num) FROM clinical_documents;",
            "expected": 0
        },
        {
            "category": "Uniqueness Constraints",
            "name": "Duplicate Chunk Index per Document",
            "sql": "SELECT COUNT(*) FROM (SELECT document_id, chunk_index, COUNT(*) FROM document_chunks GROUP BY document_id, chunk_index HAVING COUNT(*) > 1) dups;",
            "expected": 0
        },

        # 3. Chunk Sequencing & Length Invariants
        {
            "category": "Chunk Consistency",
            "name": "Chunk Character Length Mismatches",
            "sql": "SELECT COUNT(*) FROM document_chunks WHERE char_length != LENGTH(chunk_text);",
            "expected": 0
        },
        {
            "category": "Chunk Consistency",
            "name": "Empty or Degenerate Chunks",
            "sql": "SELECT COUNT(*) FROM document_chunks WHERE char_length = 0 OR token_count = 0;",
            "expected": 0
        },
        {
            "category": "Chunk Consistency",
            "name": "Non-Zero First Chunk Index",
            "sql": """
                SELECT COUNT(*) FROM (
                    SELECT document_id, MIN(chunk_index) AS min_idx 
                    FROM document_chunks 
                    GROUP BY document_id 
                    HAVING MIN(chunk_index) != 0
                ) non_zero;
            """,
            "expected": 0
        },
        {
            "category": "Chunk Consistency",
            "name": "Discontiguous Chunk Sequence",
            "sql": """
                SELECT COUNT(*) FROM (
                    SELECT document_id, MAX(chunk_index) - MIN(chunk_index) + 1 AS range_span, COUNT(*) AS actual_cnt
                    FROM document_chunks
                    GROUP BY document_id
                    HAVING (MAX(chunk_index) - MIN(chunk_index) + 1) != COUNT(*)
                ) gaps;
            """,
            "expected": 0
        },

        # 4. Clinical Temporal Invariants
        {
            "category": "Temporal Invariants",
            "name": "Discharge Preceding Admission",
            "sql": "SELECT COUNT(*) FROM encounters WHERE discharge_date IS NOT NULL AND discharge_date < admission_date;",
            "expected": 0
        },
        {
            "category": "Temporal Invariants",
            "name": "Encounter Preceding Patient Birth",
            "sql": """
                SELECT COUNT(*) FROM encounters e
                JOIN patients p ON e.patient_id = p.patient_id
                WHERE e.admission_date::date < p.date_of_birth;
            """,
            "expected": 0
        },

        # 5. Provenance Invariants
        {
            "category": "Provenance Tracking",
            "name": "Documents Missing Provenance Hash",
            "sql": """
                SELECT COUNT(*) FROM clinical_documents
                WHERE metadata->'provenance'->>'source_sha256' IS NULL;
            """,
            "expected": 0
        }
    ]

    all_passed = True
    current_category = None

    with get_connection() as conn:
        with conn.cursor() as cur:
            for check in checks:
                if check["category"] != current_category:
                    current_category = check["category"]
                    print(f"\n--- {current_category.upper()} ---")

                cur.execute(check["sql"])
                res = cur.fetchone()[0]
                expected = check["expected"]

                if res == expected:
                    status_str = "[ PASS ]"
                    print(f" {status_str} {check['name']:<48} : {res} (Expected: {expected})")
                else:
                    status_str = "[!FAIL!]"
                    all_passed = False
                    print(f" {status_str} {check['name']:<48} : {res} (Expected: {expected})")

    print("\n" + "=" * 84)
    if all_passed:
        print(" [V] ALL INGESTION INTEGRITY AND REFERENTIAL CHECKS PASSED.")
    else:
        print(" [!] ONE OR MORE INTEGRITY CHECKS FAILED! INVESTIGATE DATA INVARIANTS.")
    print("=" * 84 + "\n")

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Audit database ingestion data integrity invariants")
    args = parser.parse_args()

    success = run_integrity_audit()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
