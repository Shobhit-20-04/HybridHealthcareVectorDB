"""
Unit Tests for Data Validation, Duplicate Detection, and Provenance Stamping
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
import uuid
from datetime import date, datetime
from src.ingestion.config import GeneratorConfig
from src.ingestion.generator import ConfigurableHealthcareGenerator
from src.ingestion.validator import (
    HealthcareRecordValidator,
    IngestionSanitizer,
    ValidationError,
    compute_record_hash
)

def test_valid_cohort_passes_sanitizer():
    """Verify that a validly generated cohort passes validation cleanly."""
    config = GeneratorConfig(num_patients=5, seed=42)
    generator = ConfigurableHealthcareGenerator(config)
    cohort = generator.generate_cohort()

    sanitizer = IngestionSanitizer()
    clean_cohort, manifest = sanitizer.process_and_validate_cohort(cohort, seed=42)

    assert len(clean_cohort["patients"]) == 5
    assert manifest["status"] == "COMPLETED"
    assert manifest["seed"] == 42
    # Verify provenance attached to documents
    for doc in clean_cohort["clinical_documents"]:
        assert "provenance" in doc["metadata"]
        assert "source_sha256" in doc["metadata"]["provenance"]
        assert doc["metadata"]["provenance"]["generator_seed"] == 42

def test_invalid_uuid_raises_validation_error():
    """Verify that malformed UUIDs are caught."""
    bad_patient = {
        "patient_id": "not-a-valid-uuid",
        "synthetic_mrn": "SYN-MRN-000001",
        "gender": "M",
        "date_of_birth": "1980-01-01"
    }
    with pytest.raises(ValidationError, match="Invalid patient_id UUID"):
        HealthcareRecordValidator.validate_patient(bad_patient)

def test_invalid_mrn_format_raises_validation_error():
    """Verify synthetic MRN pattern enforcement."""
    bad_patient = {
        "patient_id": str(uuid.uuid4()),
        "synthetic_mrn": "REAL-HOSPITAL-12345",
        "gender": "F",
        "date_of_birth": "1992-05-15"
    }
    with pytest.raises(ValidationError, match="Invalid synthetic_mrn format"):
        HealthcareRecordValidator.validate_patient(bad_patient)

def test_invalid_gender_raises_validation_error():
    """Verify gender domain constraint enforcement."""
    bad_patient = {
        "patient_id": str(uuid.uuid4()),
        "synthetic_mrn": "SYN-MRN-000002",
        "gender": "INVALID_GENDER",
        "date_of_birth": "1990-01-01"
    }
    with pytest.raises(ValidationError, match="Invalid gender"):
        HealthcareRecordValidator.validate_patient(bad_patient)

def test_invalid_dates_order_raises_validation_error():
    """Verify temporal order validation: discharge before admission."""
    p_id = str(uuid.uuid4())
    bad_encounter = {
        "encounter_id": str(uuid.uuid4()),
        "patient_id": p_id,
        "encounter_type": "inpatient",
        "department": "Cardiology",
        "admission_date": "2024-05-10T14:00:00",
        "discharge_date": "2024-05-09T10:00:00",  # Prior to admission!
        "attending_clinician_id": str(uuid.uuid4()),
        "created_at": "2024-05-10T14:00:00"
    }
    with pytest.raises(ValidationError, match="precedes admission date"):
        HealthcareRecordValidator.validate_encounter(bad_encounter, known_patients={p_id})

def test_foreign_key_orphan_detection():
    """Verify foreign key constraint: encounter referencing unknown patient."""
    bad_encounter = {
        "encounter_id": str(uuid.uuid4()),
        "patient_id": str(uuid.uuid4()),  # Not in known_patients
        "encounter_type": "emergency",
        "department": "Emergency",
        "admission_date": "2024-01-01T10:00:00",
        "attending_clinician_id": str(uuid.uuid4())
    }
    with pytest.raises(ValidationError, match="Foreign key violation"):
        HealthcareRecordValidator.validate_encounter(bad_encounter, known_patients=set())

def test_invalid_icd10_code_raises_validation_error():
    """Verify ICD-10 format validation."""
    p_id = str(uuid.uuid4())
    e_id = str(uuid.uuid4())
    bad_diag = {
        "diagnosis_id": str(uuid.uuid4()),
        "encounter_id": e_id,
        "patient_id": p_id,
        "icd10_code": "NOT_AN_ICD10",
        "diagnosis_description": "Fake diagnosis"
    }
    with pytest.raises(ValidationError, match="Invalid ICD-10 code format"):
        HealthcareRecordValidator.validate_diagnosis(bad_diag, known_encounters={e_id}, known_patients={p_id})

def test_duplicate_deduplication():
    """Verify IngestionSanitizer drops duplicate records with identical business keys."""
    p_id = str(uuid.uuid4())
    patient = {
        "patient_id": p_id,
        "synthetic_mrn": "SYN-MRN-999999",
        "gender": "F",
        "date_of_birth": "1985-06-20",
        "blood_type": "O+",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00"
    }
    cohort = {
        "patients": [patient, patient.copy()],  # Exact duplicate
        "encounters": [],
        "diagnoses": [],
        "clinical_documents": [],
        "document_chunks": [],
        "medical_images": []
    }
    sanitizer = IngestionSanitizer()
    clean_cohort, manifest = sanitizer.process_and_validate_cohort(cohort, seed=42)
    assert len(clean_cohort["patients"]) == 1
    assert manifest["duplicates_detected"] == 1

def test_provenance_hash_deterministic():
    """Verify SHA-256 record hash is deterministic."""
    rec1 = {"title": "Discharge Summary", "author": "Dr. Smith", "content": "Patient recovering."}
    rec2 = {"content": "Patient recovering.", "author": "Dr. Smith", "title": "Discharge Summary"}
    assert compute_record_hash(rec1) == compute_record_hash(rec2)
