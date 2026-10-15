"""
Command-Line Retrieval-Augmented Generation (CLI-RAG) Demonstration
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Terminal-based headless CLI demonstration of hybrid clinical information retrieval.
Queries the unified database architecture using semantic vectors + relational constraints.
NO WEB UI OR CHATBOT: Strictly command-line execution and academic output.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.embeddings.text_embedder import TextEmbedder
from src.retrieval.hybrid_search import HybridSearchEngine
from src.security.rbac import SecurityContext, ClinicalRole

def run_cli_rag(
    query_text: str,
    department: str = None,
    encounter_type: str = None,
    role: str = "attending_physician",
    top_k: int = 5
):
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "synthetic"

    chunk_emb_file = data_dir / "chunk_embeddings.json"
    doc_chunks_file = data_dir / "document_chunks.json"
    docs_file = data_dir / "clinical_documents.json"
    encounters_file = data_dir / "encounters.json"
    patients_file = data_dir / "patients.json"

    if not chunk_emb_file.exists():
        print("[!] Synthetic dataset not found. Generating now...")
        from src.synthetic_generator import SyntheticHealthcareGenerator
        gen = SyntheticHealthcareGenerator()
        cohort = gen.generate_cohort(50)
        gen.save_to_disk(cohort, data_dir)
        from src.embeddings.ingest_embeddings import generate_and_save_embeddings
        generate_and_save_embeddings(data_dir)

    with open(chunk_emb_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)
    with open(doc_chunks_file, "r", encoding="utf-8") as f:
        chunks = {c["chunk_id"]: c for c in json.load(f)}
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = {d["document_id"]: d for d in json.load(f)}
    with open(encounters_file, "r", encoding="utf-8") as f:
        encounters = {e["encounter_id"]: e for e in json.load(f)}
    with open(patients_file, "r", encoding="utf-8") as f:
        patients = {p["patient_id"]: p for p in json.load(f)}

    # Join in-memory corpus
    corpus = []
    for emb in embeddings:
        cid = emb["chunk_id"]
        c = chunks.get(cid, {})
        doc = docs.get(c.get("document_id"), {})
        enc = encounters.get(doc.get("encounter_id"), {})
        p = patients.get(enc.get("patient_id"), {})

        corpus.append({
            "embedding_id": emb["embedding_id"],
            "chunk_id": cid,
            "chunk_text": c.get("chunk_text", ""),
            "title": doc.get("title", "Clinical Note"),
            "document_type": doc.get("document_type", ""),
            "department": enc.get("department", "General"),
            "encounter_type": enc.get("encounter_type", ""),
            "synthetic_mrn": p.get("synthetic_mrn", "UNKNOWN"),
            "embedding": emb["embedding"]
        })

    # Security context enforcement (RLS Simulation)
    clinical_role = ClinicalRole(role)
    sec_ctx = SecurityContext(user_id="cli-user-1", role=clinical_role, department=department)
    filtered_corpus = sec_ctx.filter_documents_offline(corpus)

    print("\n" + "=" * 78)
    print("  CLI CLINICAL RETRIEVAL DEMONSTRATION (Hybrid Relational + Vector)")
    print("=" * 78)
    print(f"[*] Query Phrase      : '{query_text}'")
    print(f"[*] Active User Role  : {clinical_role.value}")
    if department:
        print(f"[*] Relational Filter : department = '{department}'")
    if encounter_type:
        print(f"[*] Relational Filter : encounter_type = '{encounter_type}'")
    print(f"[*] Accessible Corpus : {len(filtered_corpus)} records (after RLS evaluation)")

    # Generate query embedding
    text_embedder = TextEmbedder()
    q_vec = np.array(text_embedder.embed_text(query_text), dtype=np.float32)

    # Execute Hybrid Pre-Filtered Search
    hybrid_engine = HybridSearchEngine(metric="cosine")
    filters = {}
    if department:
        filters["department"] = department
    if encounter_type:
        filters["encounter_type"] = encounter_type

    results = hybrid_engine.pre_filter_search(q_vec, filtered_corpus, filters=filters, top_k=top_k)

    if not results:
        print("\n[!] No matching clinical records found satisfying both criteria.")
        return

    print(f"\n[+] TOP-{len(results)} RETRIEVED CLINICAL PASSAGES:")
    print("-" * 78)
    for idx, r in enumerate(results, 1):
        cos_dist = r["distance"]
        cos_sim = 1.0 - cos_dist
        print(f"[{idx}] Similarity Score: {cos_sim:.4f} (Distance: {cos_dist:.4f})")
        print(f"    Patient MRN : {r.get('synthetic_mrn')} | Department: {r.get('department')} | Setting: {r.get('encounter_type')}")
        print(f"    Document    : {r.get('title')} ({r.get('document_type')})")
        print(f"    Passage     : \"{r.get('chunk_text')[:180]}...\"")
        print()

    # Formulate Clinical Synthesis Note (RAG Context Grounding)
    print("=" * 78)
    print("  DATABASE-GROUNDED CLINICAL CONTEXT SUMMARY")
    print("=" * 78)
    print("Context synthesized from verified retrieved records:")
    departments = list(set(r.get('department') for r in results))
    mrns = list(set(r.get('synthetic_mrn') for r in results))
    print(f"- Total Matching Cohort Size: {len(results)} distinct clinical passages")
    print(f"- Correlated Clinical Departments: {', '.join(departments)}")
    print(f"- Associated Synthetic Patient Profiles: {', '.join(mrns)}")
    print(f"- Top Semantic Finding: \"{results[0].get('chunk_text')[:160]}\"")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Command-Line RAG Clinical Query Runner")
    parser.add_argument("--query", type=str, default="Patient presenting with acute shortness of breath and chest pain", help="Clinical query text")
    parser.add_argument("--department", type=str, default="Cardiology", help="Relational department predicate")
    parser.add_argument("--role", type=str, default="attending_physician", choices=["clinical_admin", "attending_physician", "radiologist", "clinical_researcher"])
    parser.add_argument("--top_k", type=int, default=3, help="Top K results")
    args = parser.parse_args()

    run_cli_rag(
        query_text=args.query,
        department=args.department,
        role=args.role,
        top_k=args.top_k
    )
