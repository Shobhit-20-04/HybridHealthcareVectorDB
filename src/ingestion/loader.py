"""
Database Ingestion Loader with Parameterized SQL & Transactions
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Loads validated clinical cohorts into PostgreSQL using strict foreign key ordering,
parameterized queries, atomic transactions, and duplicate conflict handling.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from database.connection import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion.loader")

class IngestionLoaderError(Exception):
    """Raised when an ingestion loading operation fails."""
    pass

class DatabaseLoader:
    """Handles transactional, parameterized loading of clinical data cohorts."""

    def __init__(self, on_duplicate: str = "update"):
        if on_duplicate not in ("update", "ignore"):
            raise ValueError(f"Invalid on_duplicate mode: '{on_duplicate}'. Choose 'update' or 'ignore'.")
        self.on_duplicate = on_duplicate

    def truncate_tables(self) -> None:
        """Truncates all clinical data tables to prepare for a clean ingestion reload."""
        tables = [
            "image_embeddings",
            "medical_images",
            "chunk_embeddings",
            "document_chunks",
            "clinical_documents",
            "diagnoses",
            "encounters",
            "patients"
        ]
        with get_connection(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {', '.join(tables)} CASCADE;")
        logger.info("[V] Successfully truncated all clinical data tables.")

    def load_cohort(self, cohort: Dict[str, List[Dict[str, Any]]], dry_run: bool = False) -> Dict[str, int]:
        """
        Inserts all clinical entities within an atomic PostgreSQL transaction.
        Rolls back completely if any foreign key or constraint fails.
        If dry_run is True, executes all statements to test constraints, then rolls back.
        """
        patients = cohort.get("patients", [])
        encounters = cohort.get("encounters", [])
        diagnoses = cohort.get("diagnoses", [])
        documents = cohort.get("clinical_documents", [])
        chunks = cohort.get("document_chunks", [])
        images = cohort.get("medical_images", [])

        logger.info(
            f"Initiating cohort transaction ({'DRY-RUN' if dry_run else 'PERSIST'}): "
            f"{len(patients)} patients, {len(encounters)} encounters, {len(documents)} documents, "
            f"{len(chunks)} chunks, {len(images)} images."
        )

        counts = {
            "patients": 0,
            "encounters": 0,
            "diagnoses": 0,
            "clinical_documents": 0,
            "document_chunks": 0,
            "medical_images": 0
        }

        # Use non-autocommit connection for atomic multi-table transaction
        with get_connection(autocommit=False) as conn:
            try:
                with conn.cursor() as cur:
                    # -------------------------------------------------------------
                    # 1. Insert Patients
                    # -------------------------------------------------------------
                    if patients:
                        logger.info(f"Loading {len(patients)} patients...")
                        if self.on_duplicate == "update":
                            patient_stmt = """
                                INSERT INTO patients (
                                    patient_id, synthetic_mrn, gender, date_of_birth, blood_type, created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_mrn) DO UPDATE SET
                                    gender = EXCLUDED.gender,
                                    date_of_birth = EXCLUDED.date_of_birth,
                                    blood_type = EXCLUDED.blood_type,
                                    updated_at = CURRENT_TIMESTAMP;
                            """
                        else:
                            patient_stmt = """
                                INSERT INTO patients (
                                    patient_id, synthetic_mrn, gender, date_of_birth, blood_type, created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_mrn) DO NOTHING;
                            """
                        patient_data = [
                            (
                                p["patient_id"], p["synthetic_mrn"], p["gender"],
                                p["date_of_birth"], p.get("blood_type"),
                                p["created_at"], p["updated_at"]
                            )
                            for p in patients
                        ]
                        cur.executemany(patient_stmt, patient_data)
                        counts["patients"] = len(patient_data)

                    # -------------------------------------------------------------
                    # 2. Insert Encounters
                    # -------------------------------------------------------------
                    if encounters:
                        logger.info(f"Loading {len(encounters)} encounters...")
                        if self.on_duplicate == "update":
                            encounter_stmt = """
                                INSERT INTO encounters (
                                    encounter_id, patient_id, synthetic_encounter_num, encounter_type,
                                    department, admission_date, discharge_date, attending_clinician_id, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_encounter_num) DO UPDATE SET
                                    encounter_type = EXCLUDED.encounter_type,
                                    department = EXCLUDED.department,
                                    admission_date = EXCLUDED.admission_date,
                                    discharge_date = EXCLUDED.discharge_date,
                                    attending_clinician_id = EXCLUDED.attending_clinician_id;
                            """
                        else:
                            encounter_stmt = """
                                INSERT INTO encounters (
                                    encounter_id, patient_id, synthetic_encounter_num, encounter_type,
                                    department, admission_date, discharge_date, attending_clinician_id, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_encounter_num) DO NOTHING;
                            """
                        encounter_data = [
                            (
                                e["encounter_id"], e["patient_id"], e["synthetic_encounter_num"],
                                e["encounter_type"], e["department"], e["admission_date"],
                                e.get("discharge_date"), e["attending_clinician_id"], e["created_at"]
                            )
                            for e in encounters
                        ]
                        cur.executemany(encounter_stmt, encounter_data)
                        counts["encounters"] = len(encounter_data)

                    # -------------------------------------------------------------
                    # 3. Insert Diagnoses
                    # -------------------------------------------------------------
                    if diagnoses:
                        logger.info(f"Loading {len(diagnoses)} diagnoses...")
                        diag_stmt = """
                            INSERT INTO diagnoses (
                                diagnosis_id, encounter_id, patient_id, icd10_code,
                                diagnosis_description, is_primary, diagnosed_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (diagnosis_id) DO NOTHING;
                        """
                        diag_data = [
                            (
                                d["diagnosis_id"], d["encounter_id"], d["patient_id"],
                                d["icd10_code"], d["diagnosis_description"],
                                d.get("is_primary", False), d["diagnosed_at"]
                            )
                            for d in diagnoses
                        ]
                        cur.executemany(diag_stmt, diag_data)
                        counts["diagnoses"] = len(diag_data)

                    # -------------------------------------------------------------
                    # 4. Insert Clinical Documents
                    # -------------------------------------------------------------
                    if documents:
                        logger.info(f"Loading {len(documents)} clinical documents...")
                        if self.on_duplicate == "update":
                            doc_stmt = """
                                INSERT INTO clinical_documents (
                                    document_id, encounter_id, patient_id, synthetic_document_num,
                                    document_type, author_id, title, full_text, metadata, status,
                                    created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_document_num) DO UPDATE SET
                                    title = EXCLUDED.title,
                                    full_text = EXCLUDED.full_text,
                                    metadata = EXCLUDED.metadata,
                                    status = EXCLUDED.status,
                                    updated_at = CURRENT_TIMESTAMP;
                            """
                        else:
                            doc_stmt = """
                                INSERT INTO clinical_documents (
                                    document_id, encounter_id, patient_id, synthetic_document_num,
                                    document_type, author_id, title, full_text, metadata, status,
                                    created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_document_num) DO NOTHING;
                            """
                        doc_data = [
                            (
                                doc["document_id"], doc["encounter_id"], doc["patient_id"],
                                doc["synthetic_document_num"], doc["document_type"],
                                doc["author_id"], doc["title"], doc["full_text"],
                                json.dumps(doc.get("metadata", {})), doc.get("status", "signed"),
                                doc["created_at"], doc["updated_at"]
                            )
                            for doc in documents
                        ]
                        cur.executemany(doc_stmt, doc_data)
                        counts["clinical_documents"] = len(doc_data)

                    # -------------------------------------------------------------
                    # 5. Insert Document Chunks
                    # -------------------------------------------------------------
                    if chunks:
                        logger.info(f"Loading {len(chunks)} document chunks...")
                        chunk_stmt = """
                            INSERT INTO document_chunks (
                                chunk_id, document_id, chunk_index, chunk_text,
                                token_count, char_length, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (document_id, chunk_index) DO NOTHING;
                        """
                        chunk_data = [
                            (
                                c["chunk_id"], c["document_id"], c["chunk_index"],
                                c["chunk_text"], c["token_count"], c["char_length"],
                                c["created_at"]
                            )
                            for c in chunks
                        ]
                        cur.executemany(chunk_stmt, chunk_data)
                        counts["document_chunks"] = len(chunk_data)

                    # -------------------------------------------------------------
                    # 6. Insert Medical Images
                    # -------------------------------------------------------------
                    if images:
                        logger.info(f"Loading {len(images)} medical images...")
                        if self.on_duplicate == "update":
                            image_stmt = """
                                INSERT INTO medical_images (
                                    image_id, patient_id, encounter_id, document_id,
                                    synthetic_image_num, modality, body_site, synthetic_file_path,
                                    image_format, file_checksum_sha256, file_size_bytes, is_verified_asset,
                                    metadata, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_image_num) DO UPDATE SET
                                    modality = EXCLUDED.modality,
                                    body_site = EXCLUDED.body_site,
                                    synthetic_file_path = EXCLUDED.synthetic_file_path,
                                    file_checksum_sha256 = EXCLUDED.file_checksum_sha256,
                                    file_size_bytes = EXCLUDED.file_size_bytes,
                                    is_verified_asset = EXCLUDED.is_verified_asset,
                                    metadata = EXCLUDED.metadata;
                            """
                        else:
                            image_stmt = """
                                INSERT INTO medical_images (
                                    image_id, patient_id, encounter_id, document_id,
                                    synthetic_image_num, modality, body_site, synthetic_file_path,
                                    image_format, file_checksum_sha256, file_size_bytes, is_verified_asset,
                                    metadata, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (synthetic_image_num) DO NOTHING;
                            """
                        image_data = []
                        for img in images:
                            meta = img.get("metadata", {})
                            checksum = img.get("file_checksum_sha256") or meta.get("file_checksum_sha256")
                            size = img.get("file_size_bytes") if img.get("file_size_bytes") is not None else meta.get("file_size_bytes")
                            verified = img.get("is_verified_asset") if img.get("is_verified_asset") is not None else meta.get("is_verified_asset", False)
                            image_data.append((
                                img["image_id"], img["patient_id"], img["encounter_id"],
                                img.get("document_id"), img["synthetic_image_num"],
                                img["modality"], img["body_site"], img["synthetic_file_path"],
                                img["image_format"], checksum, size, verified,
                                json.dumps(meta), img["created_at"]
                            ))
                        cur.executemany(image_stmt, image_data)
                        counts["medical_images"] = len(image_data)

                if dry_run:
                    conn.rollback()
                    logger.info("[!] DRY-RUN completed: transaction rolled back safely without writing to disk.")
                else:
                    logger.info(f"[V] Successfully committed transaction for {sum(counts.values())} total rows.")

            except Exception as e:
                conn.rollback()
                logger.error(f"[X] Transaction failed and rolled back completely: {e}")
                raise

        return counts

    def load_from_directory(self, data_dir: Path, dry_run: bool = False) -> Dict[str, int]:
        """
        Loads all entity JSON files present in data_dir into the database.
        """
        cohort: Dict[str, List[Dict[str, Any]]] = {}
        entities = [
            "patients",
            "encounters",
            "diagnoses",
            "clinical_documents",
            "document_chunks",
            "medical_images"
        ]

        logger.info(f"Reading cohort entities from directory: {data_dir}")
        for entity in entities:
            file_path = data_dir / f"{entity}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    cohort[entity] = json.load(f)
                logger.info(f"  -> Found {len(cohort[entity])} records in {file_path.name}")
            else:
                cohort[entity] = []

        total_records = sum(len(v) for v in cohort.values())
        if total_records == 0:
            raise IngestionLoaderError(f"No clinical entity JSON files found in {data_dir}")

        return self.load_cohort(cohort, dry_run=dry_run)
