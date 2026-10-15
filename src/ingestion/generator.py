"""
Configurable & Deterministic Healthcare Data Generator
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Generates synthetic patients, encounters, diagnoses, clinical notes, chunks, and image metadata.
Zero real PHI. Uses deterministic PRNG seeding for reproducible benchmarks and test fixtures.
"""

import json
import random
import uuid
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Any, Tuple

from src.ingestion.config import GeneratorConfig

DEPARTMENTS = [
    "Cardiology", "Neurology", "Oncology", "Pulmonology", 
    "Emergency", "Endocrinology", "Nephrology", "Gastroenterology"
]
ENCOUNTER_TYPES = ["inpatient", "outpatient", "emergency", "telehealth"]
GENDERS = ["M", "F", "OTHER", "UNKNOWN"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

ICD10_CATALOG = [
    ("I25.10", "Atherosclerotic heart disease of native coronary artery", "Cardiology"),
    ("I50.9", "Heart failure, unspecified", "Cardiology"),
    ("I48.91", "Unspecified atrial fibrillation", "Cardiology"),
    ("E11.9", "Type 2 diabetes mellitus without complications", "Endocrinology"),
    ("E10.65", "Type 1 diabetes mellitus with hyperglycemia", "Endocrinology"),
    ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation", "Pulmonology"),
    ("J18.9", "Pneumonia, unspecified organism", "Pulmonology"),
    ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung", "Oncology"),
    ("C50.919", "Malignant neoplasm of unspecified site of female breast", "Oncology"),
    ("I63.9", "Cerebral infarction, unspecified (Ischemic stroke)", "Neurology"),
    ("G40.909", "Epilepsy, unspecified, not intractable", "Neurology"),
    ("N18.6", "End stage renal disease", "Nephrology"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis", "Gastroenterology"),
    ("K50.90", "Crohn's disease, unspecified, without complications", "Gastroenterology")
]

CLINICAL_TEMPLATES = {
    "discharge_summary": [
        "CHIEF COMPLAINT: Progressive dyspnea on exertion and bilateral lower extremity edema.\n"
        "HISTORY OF PRESENT ILLNESS: Patient presented with 3-day history of worsening shortness of breath. "
        "Cardiac history notable for ischemic cardiomyopathy with baseline left ventricular ejection fraction of 35-40%. "
        "Denies chest pain, lightheadedness, or syncope. Vitals on admission showed elevated blood pressure.\n"
        "HOSPITAL COURSE: Patient admitted to Cardiology service. Initiated on aggressive intravenous furosemide diuresis. "
        "Net negative 3.8 liters over 48 hours with dramatic symptomatic improvement in orthopnea and peripheral edema. "
        "Serum creatinine remained stable at 1.1 mg/dL. Potassium maintained > 4.0 mEq/L with oral repletion.\n"
        "DISCHARGE MEDICATIONS: Furosemide 40 mg PO daily, Lisinopril 20 mg PO daily, Metoprolol succinate 50 mg PO daily.\n"
        "DISCHARGE CONDITION & INSTRUCTIONS: Stable, ambulating on room air. Outpatient cardiology follow-up in 10 days.",

        "CHIEF COMPLAINT: Severe diabetic ketoacidosis and lethargy.\n"
        "HISTORY OF PRESENT ILLNESS: Patient with Type 1 Diabetes Mellitus presented after missing several insulin doses due to viral gastroenteritis. "
        "Laboratory evaluation demonstrated blood glucose 480 mg/dL, arterial pH 7.21, serum bicarbonate 11 mEq/L, and elevated anion gap of 22.\n"
        "HOSPITAL COURSE: Admitted to ICU. Managed with regular insulin continuous infusion protocol and isotonic crystalloid hydration. "
        "Anion gap normalized within 18 hours. Blood glucose was safely titrated to 140-180 mg/dL. Transitioned to subcutaneous glargine and lispro.\n"
        "DISCHARGE INSTRUCTIONS: Re-education on sick-day management and urine ketone testing. Follow up with Endocrinology.",

        "CHIEF COMPLAINT: Fever, productive cough, and pleuritic chest pain.\n"
        "HISTORY OF PRESENT ILLNESS: Patient presented with 4 days of chills, productive cough with purulent sputum, and right-sided pleuritic chest discomfort. "
        "Chest imaging confirmed dense right lower lobe airspace consolidation.\n"
        "HOSPITAL COURSE: Admitted to Pulmonology. Sputum culture yielded Streptococcus pneumoniae susceptible to ceftriaxone. "
        "Completed 5 days of intravenous antibiotics followed by transition to oral amoxicillin-clavulanate. Supplemental oxygen weaned off.\n"
        "DISCHARGE PLAN: Complete 5 additional days of oral antibiotics. Repeat chest radiograph in 6 weeks."
    ],
    "progress_note": [
        "SUBJECTIVE: Patient ambulated 200 feet in corridor this morning without dyspnea. Appetite is good, bowel movements regular.\n"
        "OBJECTIVE: Temperature 98.4 F, BP 118/74, Pulse 68 regular, SpO2 99% room air. Lungs clear, no wheezes or crackles. Heart regular rhythm without murmur.\n"
        "ASSESSMENT: Congestive heart failure exacerbation resolved. Normal volume status restored.\n"
        "PLAN: Discharge home this afternoon. Follow-up lab panel in 1 week.",

        "SUBJECTIVE: Reports mild numbness in bilateral toes, stable. Denies dizziness or hypoglycemic episodes.\n"
        "OBJECTIVE: Fasting morning capillary blood glucose 122 mg/dL. Sensory exam shows decreased vibration sense at bilateral great toes.\n"
        "ASSESSMENT: Stable Type 2 Diabetes on metformin and semaglutide. Mild peripheral sensory neuropathy.\n"
        "PLAN: Continue current oral hypoglycemic regimen. Annual podiatry evaluation."
    ],
    "radiology_report": [
        "EXAMINATION: Chest Radiograph (PA and Lateral views).\n"
        "CLINICAL INDICATION: Evaluate for congestive heart failure and infiltrate in patient with dyspnea.\n"
        "FINDINGS: The cardiac silhouette is mildly enlarged, consistent with mild cardiomegaly. Pulmonary vasculature shows mild vascular redistribution. "
        "No focal parenchymal consolidation, pneumothorax, or large pleural effusion. Visualized osseous structures are unremarkable.\n"
        "IMPRESSION: Mild cardiomegaly and mild pulmonary venous congestion. No acute lobar infiltrate.",

        "EXAMINATION: Non-contrast Computed Tomography of the Head.\n"
        "CLINICAL INDICATION: Acute altered mental status and left facial droop.\n"
        "FINDINGS: Symmetric brain parenchyma with normal grey-white differentiation. No acute intracranial hemorrhage or mass effect. "
        "Mild chronic periventricular white matter ischemic changes compatible with small vessel disease. Ventricles and basal cisterns are patent.\n"
        "IMPRESSION: No acute intracranial abnormality or acute territorial infarction."
    ],
    "pathology_report": [
        "CLINICAL HISTORY: Endoscopic mucosal biopsy of terminal ileum.\n"
        "GROSS DESCRIPTION: Received in formalin labeled 'terminal ileum biopsy' are three fragments of pink-tan soft tissue measuring up to 0.4 cm.\n"
        "MICROSCOPIC DESCRIPTION: Sections show small intestinal mucosa with focal crypt architectural distortion, marked chronic mucosal inflammation, "
        "and non-caseating epithelioid granulomas in the lamina propria. No viral inclusions or malignancy.\n"
        "DIAGNOSIS: Terminal ileum, biopsy: Chronic active ileitis with non-caseating granulomas, consistent with Crohn's disease."
    ],
    "consult_note": [
        "REASON FOR CONSULTATION: Management of resistant hypertension in the setting of chronic kidney disease Stage 3b.\n"
        "HISTORY: Patient admitted with hypertensive urgency (BP 192/108) on hydrochlorothiazide and amlodipine. "
        "Serum creatinine is 1.8 mg/dL (baseline 1.6), eGFR 38 mL/min/1.73m2. Urine albumin-to-creatinine ratio is 420 mg/g.\n"
        "RECOMMENDATIONS: 1. Discontinue hydrochlorothiazide; initiate loop diuretic (Torsemide 20 mg daily). "
        "2. Add low-dose Losartan 25 mg daily with close monitoring of potassium and renal function in 7 days. "
        "3. Outpatient Nephrology clinic appointment in 3 weeks."
    ]
}

IMAGE_MODALITIES = ["CR", "DX", "CT", "MR", "US"]
BODY_SITES = ["Chest", "Brain", "Abdomen", "Pelvis", "Musculoskeletal"]

class ConfigurableHealthcareGenerator:
    """Deterministic, configurable generator for relational and clinical text datasets."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def _generate_uuid(self, name: str = "") -> str:
        """Generates deterministic UUID v5 from business name/key, or PRNG fallback."""
        if name:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
        bits = self.rng.getrandbits(128)
        return str(uuid.UUID(int=bits, version=4))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Splits narrative into overlapping passages using configured window."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        chunk_idx = 0
        w_size = self.config.chunk_size_words
        overlap = self.config.chunk_overlap_words

        while start < len(words):
            end = min(start + w_size, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_index": chunk_idx,
                "chunk_text": chunk_str,
                "token_count": max(1, int(len(chunk_words) * 1.3)),
                "char_length": len(chunk_str)
            })
            chunk_idx += 1
            if end >= len(words):
                break
            start += max(1, w_size - overlap)

        return chunks

    def generate_cohort(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generates all synthetic entities according to configuration."""
        patients = []
        encounters = []
        diagnoses = []
        documents = []
        chunks = []
        images = []

        start_date = datetime(self.config.start_year, 1, 1)

        for p_idx in range(1, self.config.num_patients + 1):
            p_mrn = f"SYN-MRN-{p_idx:06d}"
            p_uuid = self._generate_uuid(p_mrn)
            gender = self.rng.choice(GENDERS)
            age_years = self.rng.randint(18, 88)
            dob = (start_date - timedelta(days=age_years * 365 + self.rng.randint(0, 360))).strftime("%Y-%m-%d")
            blood_type = self.rng.choice(BLOOD_TYPES)

            patients.append({
                "patient_id": p_uuid,
                "synthetic_mrn": p_mrn,
                "gender": gender,
                "date_of_birth": dob,
                "blood_type": blood_type,
                "created_at": start_date.isoformat(),
                "updated_at": start_date.isoformat()
            })

            # Encounters
            min_enc, max_enc = self.config.encounters_per_patient
            num_enc = self.rng.randint(min_enc, max_enc)

            for e_idx in range(1, num_enc + 1):
                enc_num = f"SYN-ENC-{p_idx:04d}-{e_idx:02d}"
                e_uuid = self._generate_uuid(enc_num)
                dept = self.rng.choice(DEPARTMENTS)
                enc_type = self.rng.choice(ENCOUNTER_TYPES)

                admission_dt = start_date + timedelta(days=self.rng.randint(1, 900), hours=self.rng.randint(0, 23))
                los_days = self.rng.randint(1, 10) if enc_type == "inpatient" else 0
                discharge_dt = admission_dt + timedelta(days=los_days) if los_days > 0 else admission_dt + timedelta(hours=self.rng.randint(2, 6))

                encounters.append({
                    "encounter_id": e_uuid,
                    "patient_id": p_uuid,
                    "synthetic_encounter_num": enc_num,
                    "encounter_type": enc_type,
                    "department": dept,
                    "admission_date": admission_dt.isoformat(),
                    "discharge_date": discharge_dt.isoformat(),
                    "attending_clinician_id": f"SYN-DOC-{self.rng.randint(100, 999)}",
                    "created_at": admission_dt.isoformat()
                })

                # Diagnoses
                min_diag, max_diag = self.config.diagnoses_per_encounter
                dept_diagnoses = [d for d in ICD10_CATALOG if d[2] == dept]
                pool = dept_diagnoses if dept_diagnoses else ICD10_CATALOG
                num_diag = min(len(pool), self.rng.randint(min_diag, max_diag))
                selected_diags = self.rng.sample(pool, k=num_diag)

                for d_order, (icd, desc, _) in enumerate(selected_diags):
                    diagnoses.append({
                        "diagnosis_id": self._generate_uuid(f"{enc_num}:{icd}:{d_order}"),
                        "encounter_id": e_uuid,
                        "patient_id": p_uuid,
                        "icd10_code": icd,
                        "diagnosis_description": desc,
                        "is_primary": (d_order == 0),
                        "diagnosed_at": admission_dt.isoformat()
                    })

                # Clinical Documents
                min_doc, max_doc = self.config.docs_per_encounter
                num_docs = self.rng.randint(min_doc, max_doc)

                for doc_idx in range(1, num_docs + 1):
                    doc_num = f"SYN-DOC-{p_idx:04d}-{e_idx:02d}-{doc_idx:02d}"
                    doc_uuid = self._generate_uuid(doc_num)
                    if enc_type == "inpatient":
                        doc_type = "discharge_summary" if doc_idx == 1 else "progress_note"
                    else:
                        doc_type = self.rng.choice(["progress_note", "radiology_report", "pathology_report", "consult_note"])

                    templates = CLINICAL_TEMPLATES.get(doc_type, CLINICAL_TEMPLATES["progress_note"])
                    narrative = self.rng.choice(templates)

                    meta = {
                        "synthetic_source": "rule-based-mimic-iv-template-v1",
                        "vital_signs": {
                            "systolic_bp": self.rng.randint(105, 160),
                            "diastolic_bp": self.rng.randint(62, 92),
                            "heart_rate": self.rng.randint(58, 105),
                            "spo2": self.rng.randint(93, 100)
                        },
                        "smoking_status": self.rng.choice(["never", "former", "current", "unknown"]),
                        "severity_score": self.rng.randint(1, 5)
                    }

                    documents.append({
                        "document_id": doc_uuid,
                        "encounter_id": e_uuid,
                        "patient_id": p_uuid,
                        "synthetic_document_num": doc_num,
                        "document_type": doc_type,
                        "author_id": f"SYN-MD-{self.rng.randint(10, 99)}",
                        "title": f"Clinical {doc_type.replace('_', ' ').title()} - {dept}",
                        "full_text": narrative,
                        "metadata": meta,
                        "status": "signed",
                        "created_at": discharge_dt.isoformat(),
                        "updated_at": discharge_dt.isoformat()
                    })

                    # Chunking
                    doc_chunks = self.chunk_text(narrative)
                    for c in doc_chunks:
                        chunks.append({
                            "chunk_id": self._generate_uuid(f"{doc_num}:{c['chunk_index']}"),
                            "document_id": doc_uuid,
                            "chunk_index": c["chunk_index"],
                            "chunk_text": c["chunk_text"],
                            "token_count": c["token_count"],
                            "char_length": c["char_length"],
                            "created_at": discharge_dt.isoformat()
                        })

                    # Optional Medical Image
                    if self.rng.random() < self.config.image_probability:
                        img_num = f"SYN-IMG-{p_idx:04d}-{e_idx:02d}-{doc_idx:02d}"
                        img_uuid = self._generate_uuid(img_num)
                        modality = "CR" if dept == "Pulmonology" else self.rng.choice(IMAGE_MODALITIES)
                        body_site = "Chest" if dept in ("Cardiology", "Pulmonology") else self.rng.choice(BODY_SITES)
                        images.append({
                            "image_id": img_uuid,
                            "patient_id": p_uuid,
                            "encounter_id": e_uuid,
                            "document_id": doc_uuid,
                            "synthetic_image_num": img_num,
                            "modality": modality,
                            "body_site": body_site,
                            "synthetic_file_path": f"/images/{modality.lower()}/{body_site.lower()}_{img_uuid[:8]}.png",
                            "image_format": "png",
                            "metadata": {
                                "matrix_size": [512, 512],
                                "slice_thickness_mm": 2.5,
                                "kvp": 120,
                                "exposure_mas": 80
                            },
                            "created_at": discharge_dt.isoformat()
                        })

        return {
            "patients": patients,
            "encounters": encounters,
            "diagnoses": diagnoses,
            "clinical_documents": documents,
            "document_chunks": chunks,
            "medical_images": images
        }

    def save_to_disk(self, data: Dict[str, List[Dict[str, Any]]], output_dir: Path):
        """Saves generated entities to JSON files in the specified directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, items in data.items():
            out_file = output_dir / f"{name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
            print(f"[+] Saved {len(items):>5} records to {out_file.name}")
