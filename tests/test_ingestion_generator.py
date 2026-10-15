"""
Unit Tests for Deterministic Synthetic Healthcare Generator
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
from src.ingestion.config import GeneratorConfig, PRESET_SIZES
from src.ingestion.generator import ConfigurableHealthcareGenerator
from src.ingestion.validator import VALID_DOCUMENT_TYPES

def test_generator_presets():
    """Verify that preset configurations produce the correct patient count."""
    for preset_name, expected_count in [("quick", 10), ("small", 25)]:
        config = GeneratorConfig.from_preset(preset_name, seed=100)
        generator = ConfigurableHealthcareGenerator(config)
        cohort = generator.generate_cohort()
        assert len(cohort["patients"]) == expected_count
        assert len(cohort["encounters"]) >= expected_count
        assert len(cohort["clinical_documents"]) >= expected_count
        assert len(cohort["document_chunks"]) >= len(cohort["clinical_documents"])

def test_reproducibility_identical_seeds():
    """Verify that identical PRNG seeds produce deterministic, bit-for-bit identical cohorts."""
    config1 = GeneratorConfig(num_patients=5, seed=42)
    config2 = GeneratorConfig(num_patients=5, seed=42)

    cohort1 = ConfigurableHealthcareGenerator(config1).generate_cohort()
    cohort2 = ConfigurableHealthcareGenerator(config2).generate_cohort()

    for entity in ["patients", "encounters", "diagnoses", "clinical_documents", "document_chunks", "medical_images"]:
        assert len(cohort1[entity]) == len(cohort2[entity])
        for r1, r2 in zip(cohort1[entity], cohort2[entity]):
            assert r1 == r2

def test_different_seeds_produce_different_data():
    """Verify that different seeds produce distinct patient cohorts and texts."""
    config1 = GeneratorConfig(num_patients=5, seed=101)
    config2 = GeneratorConfig(num_patients=5, seed=202)

    cohort1 = ConfigurableHealthcareGenerator(config1).generate_cohort()
    cohort2 = ConfigurableHealthcareGenerator(config2).generate_cohort()

    dobs1 = [p["date_of_birth"] for p in cohort1["patients"]]
    dobs2 = [p["date_of_birth"] for p in cohort2["patients"]]
    texts1 = [d["full_text"] for d in cohort1["clinical_documents"]]
    texts2 = [d["full_text"] for d in cohort2["clinical_documents"]]
    assert dobs1 != dobs2
    assert texts1 != texts2

def test_document_types_coverage():
    """Verify that the generator produces documents across diverse clinical document types."""
    config = GeneratorConfig(num_patients=30, seed=777)
    cohort = ConfigurableHealthcareGenerator(config).generate_cohort()

    generated_types = {d["document_type"] for d in cohort["clinical_documents"]}
    expected_types = {"discharge_summary", "progress_note", "radiology_report", "pathology_report", "consult_note"}
    assert expected_types.issubset(generated_types)

def test_chunking_structure_and_invariants():
    """Verify chunk sequence indices, lengths, and non-empty content."""
    config = GeneratorConfig(num_patients=10, seed=42, chunk_size_words=50, chunk_overlap_words=10)
    cohort = ConfigurableHealthcareGenerator(config).generate_cohort()

    chunks = cohort["document_chunks"]
    assert len(chunks) > 0

    doc_chunk_map = {}
    for c in chunks:
        doc_id = c["document_id"]
        doc_chunk_map.setdefault(doc_id, []).append(c)
        assert c["char_length"] == len(c["chunk_text"])
        assert c["token_count"] > 0
        assert len(c["chunk_text"].strip()) > 0

    # Ensure every document has contiguous chunk indices starting at 0
    for doc_id, doc_chunks in doc_chunk_map.items():
        indices = [c["chunk_index"] for c in doc_chunks]
        assert indices == list(range(len(doc_chunks)))
