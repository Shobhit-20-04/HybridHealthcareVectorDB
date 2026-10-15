"""
Integration & Unit Tests for Database Ingestion Loader
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
import uuid
from datetime import datetime
from src.ingestion.config import GeneratorConfig
from src.ingestion.generator import ConfigurableHealthcareGenerator
from src.ingestion.validator import IngestionSanitizer
from src.ingestion.loader import DatabaseLoader
from database.connection import get_connection

@pytest.fixture(scope="module")
def small_clean_cohort():
    """Generates and sanitizes a 3-patient deterministic cohort for testing."""
    config = GeneratorConfig(num_patients=3, seed=888)
    raw = ConfigurableHealthcareGenerator(config).generate_cohort()
    clean, _ = IngestionSanitizer().process_and_validate_cohort(raw, seed=888)
    yield clean
    # Teardown: delete test patients inserted during test run
    test_mrns = [p["synthetic_mrn"] for p in clean["patients"]]
    with get_connection(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM patients WHERE synthetic_mrn = ANY(%s);", (test_mrns,))

def test_loader_dry_run(small_clean_cohort):
    """Verify that dry_run executes statements and rolls back without committing rows."""
    loader = DatabaseLoader(on_duplicate="update")

    # Get count before dry run
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM patients;")
            count_before = cur.fetchone()[0]

    counts = loader.load_cohort(small_clean_cohort, dry_run=True)
    assert counts["patients"] == 3

    # Check count after dry run is completely unchanged
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM patients;")
            count_after = cur.fetchone()[0]

    assert count_before == count_after

def test_loader_live_insertion_and_foreign_keys(small_clean_cohort):
    """Verify transactional insertion into live PostgreSQL and foreign key references."""
    loader = DatabaseLoader(on_duplicate="update")
    counts = loader.load_cohort(small_clean_cohort, dry_run=False)

    assert counts["patients"] == 3
    assert counts["encounters"] > 0
    assert counts["clinical_documents"] > 0
    assert counts["document_chunks"] > 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Verify patient existence
            cur.execute("SELECT COUNT(*) FROM patients WHERE synthetic_mrn = %s;", (small_clean_cohort["patients"][0]["synthetic_mrn"],))
            assert cur.fetchone()[0] == 1

            # Verify referential join between encounters and patients
            cur.execute("""
                SELECT COUNT(*) 
                FROM encounters e 
                JOIN patients p ON e.patient_id = p.patient_id;
            """)
            assert cur.fetchone()[0] >= counts["encounters"]

def test_loader_duplicate_handling_on_conflict(small_clean_cohort):
    """Verify re-running ingestion with 'ignore' or 'update' handles conflicts gracefully."""
    loader_ignore = DatabaseLoader(on_duplicate="ignore")
    counts = loader_ignore.load_cohort(small_clean_cohort, dry_run=False)
    assert counts["patients"] == 3

    loader_update = DatabaseLoader(on_duplicate="update")
    counts_update = loader_update.load_cohort(small_clean_cohort, dry_run=False)
    assert counts_update["patients"] == 3

def test_loader_transaction_rollback_on_foreign_key_violation():
    """Verify atomic transaction rollback when an invalid foreign key constraint is encountered."""
    loader = DatabaseLoader(on_duplicate="update")

    fake_patient_id = str(uuid.uuid4())
    fake_encounter_num = f"SYN-ENC-FAIL-{uuid.uuid4().hex[:6]}"

    # Cohort with an encounter referencing a non-existent patient
    bad_cohort = {
        "patients": [],
        "encounters": [
            {
                "encounter_id": str(uuid.uuid4()),
                "patient_id": fake_patient_id,  # Invalid FK
                "synthetic_encounter_num": fake_encounter_num,
                "encounter_type": "emergency",
                "department": "Emergency",
                "admission_date": "2024-01-01T10:00:00",
                "discharge_date": "2024-01-02T10:00:00",
                "attending_clinician_id": str(uuid.uuid4()),
                "created_at": "2024-01-01T10:00:00"
            }
        ],
        "diagnoses": [],
        "clinical_documents": [],
        "document_chunks": [],
        "medical_images": []
    }

    with pytest.raises(Exception):
        loader.load_cohort(bad_cohort, dry_run=False)

    # Verify that nothing was inserted for this failed encounter
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM encounters WHERE synthetic_encounter_num = %s;", (fake_encounter_num,))
            assert cur.fetchone()[0] == 0
