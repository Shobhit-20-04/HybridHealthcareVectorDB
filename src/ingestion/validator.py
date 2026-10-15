"""
Data Validation, Provenance Stamping, and Duplicate Detection
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Enforces schema-level validation, referential checks, duplicate detection,
and cryptographic provenance tracking before database insertion.
"""

import re
import hashlib
import logging
from datetime import datetime, date
from uuid import UUID
from typing import Dict, List, Any, Tuple, Set, Optional

logger = logging.getLogger("ingestion.validator")

class ValidationError(Exception):
    """Raised when clinical data records fail integrity or domain rules."""
    pass

VALID_GENDERS = {"M", "F", "OTHER", "UNKNOWN"}
VALID_BLOOD_TYPES = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
VALID_ENCOUNTER_TYPES = {"inpatient", "outpatient", "emergency", "telehealth"}
VALID_DOCUMENT_TYPES = {"discharge_summary", "progress_note", "radiology_report", "pathology_report", "consult_note"}
VALID_DOCUMENT_STATUS = {"draft", "preliminary", "signed", "amended"}
VALID_IMAGE_MODALITIES = {"CR", "DX", "CT", "MR", "US"}
VALID_IMAGE_FORMATS = {"dicom", "png", "jpeg", "nifti"}
ICD10_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")
SYNTHETIC_MRN_PATTERN = re.compile(r"^SYN-MRN-\d{6}$")

def is_valid_uuid(val: Any) -> bool:
    try:
        UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False

def compute_record_hash(record: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash of a record's business keys."""
    sorted_items = sorted((k, str(v)) for k, v in record.items() if k not in ("created_at", "updated_at"))
    content_str = "|".join(f"{k}:{v}" for k, v in sorted_items)
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

class HealthcareRecordValidator:
    """Validates relational integrity, formats, and domain constraints."""

    @staticmethod
    def validate_patient(record: Dict[str, Any]) -> None:
        if not is_valid_uuid(record.get("patient_id")):
            raise ValidationError(f"Invalid patient_id UUID: {record.get('patient_id')}")
        mrn = record.get("synthetic_mrn", "")
        if not SYNTHETIC_MRN_PATTERN.match(mrn):
            raise ValidationError(f"Invalid synthetic_mrn format: '{mrn}'. Expected 'SYN-MRN-XXXXXX'.")
        if record.get("gender") not in VALID_GENDERS:
            raise ValidationError(f"Invalid gender '{record.get('gender')}' for patient {mrn}")
        try:
            dob = date.fromisoformat(str(record.get("date_of_birth")))
            if dob > date.today() or dob < date(1900, 1, 1):
                raise ValidationError(f"Unrealistic date_of_birth '{dob}' for patient {mrn}")
        except ValueError:
            raise ValidationError(f"Invalid ISO date_of_birth '{record.get('date_of_birth')}' for patient {mrn}")
        bt = record.get("blood_type")
        if bt is not None and bt not in VALID_BLOOD_TYPES:
            raise ValidationError(f"Invalid blood_type '{bt}' for patient {mrn}")

    @staticmethod
    def validate_encounter(record: Dict[str, Any], known_patients: Set[str]) -> None:
        if not is_valid_uuid(record.get("encounter_id")):
            raise ValidationError(f"Invalid encounter_id UUID: {record.get('encounter_id')}")
        p_id = str(record.get("patient_id"))
        if p_id not in known_patients:
            raise ValidationError(f"Foreign key violation: encounter references unknown patient_id '{p_id}'")
        if record.get("encounter_type") not in VALID_ENCOUNTER_TYPES:
            raise ValidationError(f"Invalid encounter_type '{record.get('encounter_type')}'")
        if not record.get("department"):
            raise ValidationError("Encounter department cannot be empty.")
        
        # Validate admission and discharge dates
        try:
            adm = datetime.fromisoformat(str(record.get("admission_date")))
            dis = record.get("discharge_date")
            if dis:
                dis_dt = datetime.fromisoformat(str(dis))
                if dis_dt < adm:
                    raise ValidationError(f"Discharge date '{dis}' precedes admission date '{adm}'")
        except ValueError as e:
            raise ValidationError(f"Invalid ISO timestamp in encounter: {e}")

    @staticmethod
    def validate_diagnosis(record: Dict[str, Any], known_encounters: Set[str], known_patients: Set[str]) -> None:
        if not is_valid_uuid(record.get("diagnosis_id")):
            raise ValidationError(f"Invalid diagnosis_id UUID: {record.get('diagnosis_id')}")
        if str(record.get("encounter_id")) not in known_encounters:
            raise ValidationError("Diagnosis references unknown encounter_id.")
        if str(record.get("patient_id")) not in known_patients:
            raise ValidationError("Diagnosis references unknown patient_id.")
        icd = record.get("icd10_code", "")
        if not ICD10_PATTERN.match(icd):
            raise ValidationError(f"Invalid ICD-10 code format: '{icd}'")
        if not record.get("diagnosis_description"):
            raise ValidationError("Diagnosis description cannot be empty.")

    @staticmethod
    def validate_clinical_document(record: Dict[str, Any], known_encounters: Set[str], known_patients: Set[str]) -> None:
        if not is_valid_uuid(record.get("document_id")):
            raise ValidationError(f"Invalid document_id UUID: {record.get('document_id')}")
        if str(record.get("encounter_id")) not in known_encounters:
            raise ValidationError("Clinical document references unknown encounter_id.")
        if str(record.get("patient_id")) not in known_patients:
            raise ValidationError("Clinical document references unknown patient_id.")
        if record.get("document_type") not in VALID_DOCUMENT_TYPES:
            raise ValidationError(f"Invalid document_type: '{record.get('document_type')}'")
        if record.get("status") not in VALID_DOCUMENT_STATUS:
            raise ValidationError(f"Invalid document status: '{record.get('status')}'")
        text = record.get("full_text", "")
        if not text or len(text.strip()) < 20:
            raise ValidationError("Clinical document full_text is too short or empty (min 20 characters).")

    @staticmethod
    def validate_document_chunk(record: Dict[str, Any], known_documents: Set[str]) -> None:
        if not is_valid_uuid(record.get("chunk_id")):
            raise ValidationError(f"Invalid chunk_id UUID: {record.get('chunk_id')}")
        if str(record.get("document_id")) not in known_documents:
            raise ValidationError("Document chunk references unknown document_id.")
        idx = record.get("chunk_index")
        if idx is None or idx < 0:
            raise ValidationError(f"chunk_index must be a non-negative integer, got {idx}")
        text = record.get("chunk_text", "")
        if not text:
            raise ValidationError("chunk_text cannot be empty.")
        if record.get("token_count", 0) <= 0:
            raise ValidationError("token_count must be positive.")
        if record.get("char_length") != len(text):
            raise ValidationError(f"char_length {record.get('char_length')} does not match actual length {len(text)}")

    @staticmethod
    def validate_medical_image(record: Dict[str, Any], known_encounters: Set[str], known_patients: Set[str]) -> None:
        if not is_valid_uuid(record.get("image_id")):
            raise ValidationError(f"Invalid image_id UUID: {record.get('image_id')}")
        if str(record.get("patient_id")) not in known_patients:
            raise ValidationError("Medical image references unknown patient_id.")
        if str(record.get("encounter_id")) not in known_encounters:
            raise ValidationError("Medical image references unknown encounter_id.")
        if record.get("modality") not in VALID_IMAGE_MODALITIES:
            raise ValidationError(f"Invalid image modality: '{record.get('modality')}'")
        if record.get("image_format") not in VALID_IMAGE_FORMATS:
            raise ValidationError(f"Invalid image_format: '{record.get('image_format')}'")
        if not record.get("synthetic_file_path"):
            raise ValidationError("synthetic_file_path cannot be empty.")

class IngestionSanitizer:
    """Detects duplicate records, attaches cryptographic provenance stamps, and validates cohorts."""

    def __init__(self):
        self.validator = HealthcareRecordValidator()

    def process_and_validate_cohort(
        self,
        raw_cohort: Dict[str, List[Dict[str, Any]]],
        seed: int = 42
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Validates all entities, drops duplicate business keys, and appends provenance metadata.
        Returns (sanitized_cohort, audit_metrics).
        """
        audit = {
            "status": "COMPLETED",
            "timestamp": datetime.utcnow().isoformat(),
            "random_seed": seed,
            "seed": seed,
            "provenance_algorithm": "deterministic_synthetic_mimic_v1",
            "counts_raw": {k: len(v) for k, v in raw_cohort.items()},
            "duplicates_dropped": {},
            "duplicates_detected": 0,
            "validation_errors": []
        }

        sanitized_patients = []
        seen_mrns = set()
        known_patient_ids = set()

        for p in raw_cohort.get("patients", []):
            mrn = p.get("synthetic_mrn")
            if mrn in seen_mrns:
                logger.warning(f"Duplicate patient MRN detected: {mrn}, skipping.")
                continue
            self.validator.validate_patient(p)
            seen_mrns.add(mrn)
            known_patient_ids.add(str(p["patient_id"]))
            # Attach provenance stamp
            p["metadata"] = {
                "provenance": {
                    "source": "synthetic_generator",
                    "seed": seed,
                    "record_hash": compute_record_hash(p)
                }
            }
            sanitized_patients.append(p)
        audit["duplicates_dropped"]["patients"] = len(raw_cohort.get("patients", [])) - len(sanitized_patients)

        sanitized_encounters = []
        seen_enc_nums = set()
        known_encounter_ids = set()

        for e in raw_cohort.get("encounters", []):
            enc_num = e.get("synthetic_encounter_num")
            if enc_num in seen_enc_nums:
                logger.warning(f"Duplicate encounter number: {enc_num}, skipping.")
                continue
            self.validator.validate_encounter(e, known_patient_ids)
            seen_enc_nums.add(enc_num)
            known_encounter_ids.add(str(e["encounter_id"]))
            sanitized_encounters.append(e)
        audit["duplicates_dropped"]["encounters"] = len(raw_cohort.get("encounters", [])) - len(sanitized_encounters)

        sanitized_diagnoses = []
        seen_diag_keys = set()
        for d in raw_cohort.get("diagnoses", []):
            diag_key = (str(d.get("encounter_id")), d.get("icd10_code"))
            if diag_key in seen_diag_keys:
                continue
            self.validator.validate_diagnosis(d, known_encounter_ids, known_patient_ids)
            seen_diag_keys.add(diag_key)
            sanitized_diagnoses.append(d)
        audit["duplicates_dropped"]["diagnoses"] = len(raw_cohort.get("diagnoses", [])) - len(sanitized_diagnoses)

        sanitized_docs = []
        seen_doc_nums = set()
        known_document_ids = set()

        for doc in raw_cohort.get("clinical_documents", []):
            dnum = doc.get("synthetic_document_num")
            if dnum in seen_doc_nums:
                continue
            self.validator.validate_clinical_document(doc, known_encounter_ids, known_patient_ids)
            seen_doc_nums.add(dnum)
            known_document_ids.add(str(doc["document_id"]))
            
            # Enrich metadata with document provenance
            meta = doc.get("metadata", {})
            rec_hash = compute_record_hash(doc)
            meta["provenance"] = {
                "source": "synthetic_generator",
                "generator_seed": seed,
                "document_hash": rec_hash,
                "source_sha256": rec_hash,
                "generated_at": doc.get("created_at") or datetime.utcnow().isoformat()
            }
            doc["metadata"] = meta
            sanitized_docs.append(doc)
        audit["duplicates_dropped"]["clinical_documents"] = len(raw_cohort.get("clinical_documents", [])) - len(sanitized_docs)

        sanitized_chunks = []
        seen_chunk_keys = set()
        for c in raw_cohort.get("document_chunks", []):
            chunk_key = (str(c.get("document_id")), c.get("chunk_index"))
            if chunk_key in seen_chunk_keys:
                continue
            self.validator.validate_document_chunk(c, known_document_ids)
            seen_chunk_keys.add(chunk_key)
            sanitized_chunks.append(c)
        audit["duplicates_dropped"]["document_chunks"] = len(raw_cohort.get("document_chunks", [])) - len(sanitized_chunks)

        sanitized_images = []
        seen_img_nums = set()
        for img in raw_cohort.get("medical_images", []):
            inum = img.get("synthetic_image_num")
            if inum in seen_img_nums:
                continue
            self.validator.validate_medical_image(img, known_encounter_ids, known_patient_ids)
            seen_img_nums.add(inum)
            sanitized_images.append(img)
        audit["duplicates_dropped"]["medical_images"] = len(raw_cohort.get("medical_images", [])) - len(sanitized_images)

        audit["duplicates_detected"] = sum(audit["duplicates_dropped"].values())

        audit["counts_sanitized"] = {
            "patients": len(sanitized_patients),
            "encounters": len(sanitized_encounters),
            "diagnoses": len(sanitized_diagnoses),
            "clinical_documents": len(sanitized_docs),
            "document_chunks": len(sanitized_chunks),
            "medical_images": len(sanitized_images)
        }

        clean_cohort = {
            "patients": sanitized_patients,
            "encounters": sanitized_encounters,
            "diagnoses": sanitized_diagnoses,
            "clinical_documents": sanitized_docs,
            "document_chunks": sanitized_chunks,
            "medical_images": sanitized_images
        }
        return clean_cohort, audit
