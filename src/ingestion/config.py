"""
Configuration Module for Healthcare Data Ingestion
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

PRESET_SIZES = {
    "quick": 10,       # Instant smoke test
    "small": 25,       # Fast development & testing
    "medium": 100,     # Standard integration testing
    "large": 500,      # Deep querying & indexing tests
    "benchmark": 1000  # High-throughput benchmark evaluations
}

@dataclass
class GeneratorConfig:
    """Configurable parameters for deterministic synthetic healthcare cohort generation."""
    seed: int = 42
    num_patients: int = 25
    encounters_per_patient: Tuple[int, int] = (1, 3)
    diagnoses_per_encounter: Tuple[int, int] = (1, 3)
    docs_per_encounter: Tuple[int, int] = (1, 2)
    image_probability: float = 0.4
    chunk_size_words: int = 80
    chunk_overlap_words: int = 15
    start_year: int = 2023

    @classmethod
    def from_preset(cls, preset_name: str, seed: int = 42) -> "GeneratorConfig":
        """Instantiates a configuration using standard preset scales."""
        if preset_name not in PRESET_SIZES:
            raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {list(PRESET_SIZES.keys())}")
        return cls(num_patients=PRESET_SIZES[preset_name], seed=seed)

@dataclass
class IngestionConfig:
    """Parameters governing database transaction batching and duplicate handling."""
    batch_size: int = 250
    on_duplicate: str = "update"  # "update" | "ignore" | "error"
    validate_before_load: bool = True
    dry_run: bool = False
