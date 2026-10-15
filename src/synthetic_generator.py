"""
Synthetic Healthcare Data Generator
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Generates realistic, de-identified clinical records including patients, encounters,
ICD-10 diagnoses, multi-paragraph clinical narratives, document chunks, and medical imaging metadata.
Strictly adheres to synthetic IDs (zero PHI).
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Clinical Departments & Cohorts
DEPARTMENTS = [
    "Cardiology", "Neurology", "Oncology", "Pulmonology", 
    "Emergency", "Endocrinology", "Nephrology", "Gastroenterology"
]

ENCOUNTER_TYPES = ["inpatient", "outpatient", "emergency", "telehealth"]
GENDERS = ["M", "F", "OTHER"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Common ICD-10 Diagnostic Codes with clinical descriptions
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
        "CHIEF COMPLAINT: Acute shortness of breath and substernal chest discomfort.\n"
        "HISTORY OF PRESENT ILLNESS: Patient presented to the Emergency Department with progressive dyspnea on exertion over the past 48 hours. "
        "Associated symptoms include mild orthopnea and bilateral lower extremity edema. Prior cardiac history includes coronary artery disease and hypertension.\n"
        "HOSPITAL COURSE: Patient was admitted to Cardiology. Echocardiogram revealed left ventricular ejection fraction of 40% with regional wall motion abnormalities. "
        "Intravenous diuresis was initiated with significant improvement in respiratory mechanics. Troponin levels peaked at 0.12 ng/mL and trended downward.\n"
        "DISCHARGE CONDITION: Stable on oral medical regimen. Follow-up scheduled in cardiology clinic in 2 weeks.",
        
        "CHIEF COMPLAINT: Hyperglycemia and altered mental status.\n"
        "HISTORY OF PRESENT ILLNESS: Patient with long-standing Type 2 Diabetes presented with serum glucose exceeding 450 mg/dL and severe dehydration. "
        "Urinalysis was negative for ketones. Basic metabolic panel showed serum sodium 134 mEq/L and potassium 4.8 mEq/L.\n"
        "HOSPITAL COURSE: Aggressive fluid resuscitation with isotonic saline followed by continuous intravenous regular insulin infusion. "
        "Blood glucose stabilized in the 140-180 mg/dL range. Transitioned to basal-bolus subcutaneous insulin regimen prior to discharge.\n"
        "DISCHARGE PLAN: Patient education completed on insulin administration and continuous glucose monitoring.",

        "CHIEF COMPLAINT: Productive cough and high-grade fevers.\n"
        "HISTORY OF PRESENT ILLNESS: Patient presented with 5 days of worsening productive cough with purulent sputum, pleuritic chest pain, and fever up to 102.4 F. "
        "Chest X-ray demonstrated dense right middle lobe consolidation consistent with community-acquired pneumonia.\n"
        "HOSPITAL COURSE: Treated with IV Ceftriaxone and Azithromycin. Sputum culture yielded Streptococcus pneumoniae sensitive to beta-lactams. "
        "Oxygen requirement normalized to ambient room air.\n"
        "DISCHARGE INSTRUCTIONS: Complete remaining oral antibiotic course. Return if shortness of breath recurs."
    ],
    "progress_note": [
        "SUBJECTIVE: Patient reports feeling noticeably better this morning. Denies chest pain or palpitations. Ambulating in hallway without desaturation.\n"
        "OBJECTIVE: BP 124/78, HR 72 regular, Temp 98.6 F, SpO2 98% on room air. Lungs clear to auscultation bilaterally. No peripheral edema.\n"
        "ASSESSMENT: Acute decompensated heart failure now compensated. Stable on oral furosemide, lisinopril, and metoprolol succinate.\n"
        "PLAN: Discharge today. Follow-up BMP in 7 days.",

        "SUBJECTIVE: Mild headache, denies focal weakness, numbness, or visual disturbances. Tolerating diabetic diet.\n"
        "OBJECTIVE: Vitals stable. Fasting blood sugar 132 mg/dL. Neurologic exam non-focal. Cranial nerves II-XII intact.\n"
        "ASSESSMENT: Improving glycemic control. Diabetic neuropathy stable.\n"
        "PLAN: Continue Lantus 24 units bedtime and Humalog 6 units before meals."
    ],
    "radiology_report": [
        "EXAMINATION: Chest Radiograph, PA and Lateral views.\n"
        "INDICATION: Acute dyspnea, evaluate for congestive heart failure and infiltrate.\n"
        "FINDINGS: The cardiothoracic ratio is mildly prominent, consistent with mild cardiomegaly. Prominence of pulmonary vascularity with interstitial markings in the perihilar regions. No focal pneumonic consolidation. Pleural spaces are clear without significant effusion. Bony thorax is intact.\n"
        "IMPRESSION: Mild pulmonary vascular congestion compatible with early fluid overload without discrete acute pneumonia.",

        "EXAMINATION: CT Head without intravenous contrast.\n"
        "INDICATION: Acute onset right-sided weakness and slurred speech.\n"
        "FINDINGS: No acute intracranial hemorrhage or extra-axial fluid collection. Mild age-related cerebral atrophy and chronic microvascular ischemic white matter changes. Ventricles and sulci are symmetric and within normal limits for age. Grey-white matter differentiation is preserved.\n"
        "IMPRESSION: No evidence of acute intracranial hemorrhage or large territorial territorial infarction."
    ],
    "pathology_report": [
        "CLINICAL HISTORY: Bronchoscopic biopsy, right upper lobe lesion.\n"
        "GROSS DESCRIPTION: Specimen consists of three tan-white tissue cores measuring 0.4 to 0.8 cm in length.\n"
        "MICROSCOPIC DESCRIPTION: Sections demonstrate invasive non-small cell carcinoma with glandular differentiation. Tumor cells exhibit pleomorphic nuclei, prominent nucleoli, and abundant eosinophilic cytoplasm. Immunohistochemistry is positive for TTF-1 and Napsin-A, negative for p40.\n"
        "DIAGNOSIS: Invasive adenocarcinoma of pulmonary origin."
    ]
}

IMAGE_MODALITIES = ["CR", "DX", "CT", "MR"]
BODY_SITES = ["Chest", "Brain", "Abdomen", "Pelvis", "Musculoskeletal"]

class SyntheticHealthcareGenerator:
    """Generates synthetic relational and narrative healthcare data."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed

    def chunk_text(self, text: str, chunk_size: int = 150, overlap: int = 30) -> List[Dict[str, Any]]:
        """Splits narrative into overlapping passages."""
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_index": chunk_idx,
                "chunk_text": chunk_str,
                "token_count": max(1, int(len(chunk_words) * 1.3)), # approximate BPE tokens
                "char_length": len(chunk_str)
            })
            chunk_idx += 1
            if end >= len(words):
                break
            start += (chunk_size - overlap)

        return chunks

    def generate_cohort(self, num_patients: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Generates a complete relational healthcare dataset."""
        patients = []
        encounters = []
        diagnoses = []
        documents = []
        chunks = []
        images = []

        start_date = datetime(2023, 1, 1)

        for p_idx in range(1, num_patients + 1):
            p_uuid = str(uuid.uuid4())
            p_mrn = f"SYN-MRN-{p_idx:06d}"
            gender = random.choice(GENDERS)
            age_years = random.randint(18, 88)
            dob = (start_date - timedelta(days=age_years * 365 + random.randint(0, 360))).strftime("%Y-%m-%d")
            blood_type = random.choice(BLOOD_TYPES)

            patients.append({
                "patient_id": p_uuid,
                "synthetic_mrn": p_mrn,
                "gender": gender,
                "date_of_birth": dob,
                "blood_type": blood_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })

            # Generate 1 to 4 encounters per patient
            num_encounters = random.randint(1, 4)
            for e_idx in range(1, num_encounters + 1):
                e_uuid = str(uuid.uuid4())
                enc_num = f"SYN-ENC-{p_idx:04d}-{e_idx:02d}"
                dept = random.choice(DEPARTMENTS)
                enc_type = random.choice(ENCOUNTER_TYPES)
                
                admission_dt = start_date + timedelta(days=random.randint(1, 1000), hours=random.randint(0, 23))
                length_of_stay_days = random.randint(1, 12) if enc_type == "inpatient" else 0
                discharge_dt = admission_dt + timedelta(days=length_of_stay_days) if length_of_stay_days > 0 else admission_dt + timedelta(hours=random.randint(2, 6))

                encounters.append({
                    "encounter_id": e_uuid,
                    "patient_id": p_uuid,
                    "synthetic_encounter_num": enc_num,
                    "encounter_type": enc_type,
                    "department": dept,
                    "admission_date": admission_dt.isoformat(),
                    "discharge_date": discharge_dt.isoformat(),
                    "attending_clinician_id": f"SYN-DOC-{random.randint(100, 999)}",
                    "created_at": admission_dt.isoformat()
                })

                # Assign 1 to 3 diagnoses to this encounter
                dept_diagnoses = [d for d in ICD10_CATALOG if d[2] == dept]
                chosen_diagnoses = dept_diagnoses if dept_diagnoses else ICD10_CATALOG
                selected = random.sample(chosen_diagnoses, k=min(len(chosen_diagnoses), random.randint(1, 2)))

                for d_order, (icd, desc, _) in enumerate(selected):
                    diagnoses.append({
                        "diagnosis_id": str(uuid.uuid4()),
                        "encounter_id": e_uuid,
                        "patient_id": p_uuid,
                        "icd10_code": icd,
                        "diagnosis_description": desc,
                        "is_primary": (d_order == 0),
                        "diagnosed_at": admission_dt.isoformat()
                    })

                # Generate 1 to 2 clinical documents
                num_docs = random.randint(1, 2)
                for doc_idx in range(1, num_docs + 1):
                    doc_uuid = str(uuid.uuid4())
                    doc_type = "discharge_summary" if enc_type == "inpatient" else random.choice(["progress_note", "radiology_report", "pathology_report"])
                    doc_templates = CLINICAL_TEMPLATES.get(doc_type, CLINICAL_TEMPLATES["progress_note"])
                    narrative = random.choice(doc_templates)

                    # Semistructured clinical metadata
                    meta = {
                        "synthetic_source": "MIMIC-IV-style-synthesizer",
                        "vital_signs": {
                            "systolic_bp": random.randint(100, 165),
                            "diastolic_bp": random.randint(60, 95),
                            "heart_rate": random.randint(60, 110),
                            "spo2": random.randint(92, 100)
                        },
                        "smoking_status": random.choice(["never", "former", "current", "unknown"]),
                        "severity_score": random.randint(1, 5)
                    }

                    documents.append({
                        "document_id": doc_uuid,
                        "encounter_id": e_uuid,
                        "patient_id": p_uuid,
                        "synthetic_document_num": f"SYN-DOC-{p_idx:04d}-{e_idx:02d}-{doc_idx:02d}",
                        "document_type": doc_type,
                        "author_id": f"SYN-MD-{random.randint(10, 99)}",
                        "title": f"Clinical {doc_type.replace('_', ' ').title()} - {dept}",
                        "full_text": narrative,
                        "metadata": meta,
                        "status": "signed",
                        "created_at": discharge_dt.isoformat(),
                        "updated_at": discharge_dt.isoformat()
                    })

                    # Chunk document into passages
                    doc_chunks = self.chunk_text(narrative, chunk_size=80, overlap=15)
                    for c in doc_chunks:
                        chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "document_id": doc_uuid,
                            "chunk_index": c["chunk_index"],
                            "chunk_text": c["chunk_text"],
                            "token_count": c["token_count"],
                            "char_length": c["char_length"],
                            "created_at": discharge_dt.isoformat()
                        })

                    # Optional medical image for radiology notes or cardiology/pulmonology
                    if doc_type == "radiology_report" or dept in ("Cardiology", "Pulmonology"):
                        img_uuid = str(uuid.uuid4())
                        modality = "CR" if dept == "Pulmonology" else random.choice(IMAGE_MODALITIES)
                        body_site = "Chest" if dept in ("Cardiology", "Pulmonology") else random.choice(BODY_SITES)
                        images.append({
                            "image_id": img_uuid,
                            "patient_id": p_uuid,
                            "encounter_id": e_uuid,
                            "document_id": doc_uuid,
                            "synthetic_image_num": f"SYN-IMG-{p_idx:04d}-{e_idx:02d}",
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
        """Saves generated entities to JSON files in output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, items in data.items():
            out_file = output_dir / f"{name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
            print(f"[+] Saved {len(items):>5} rows to {out_file.name}")

if __name__ == "__main__":
    generator = SyntheticHealthcareGenerator(seed=1337)
    out_path = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    print(f"[*] Generating synthetic cohort in {out_path}...")
    dataset = generator.generate_cohort(num_patients=100)
    generator.save_to_disk(dataset, out_path)
    print("[V] Synthetic cohort generation completed successfully.")
