"""
Synthetic Cohort Generation & Preprocessing CLI
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Generates deterministic synthetic cohorts, validates records, attaches cryptographic
provenance stamps, and preserves strict separation between raw, synthetic, and processed tiers.
"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.config import GeneratorConfig, PRESET_SIZES
from src.ingestion.generator import ConfigurableHealthcareGenerator
from src.ingestion.validator import IngestionSanitizer

def main():
    parser = argparse.ArgumentParser(description="Deterministic Synthetic Healthcare Data Generator")
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=list(PRESET_SIZES.keys()), 
        default="small",
        help=f"Dataset size preset: quick (10), small (25), medium (100), large (500), benchmark (1000)"
    )
    parser.add_argument("--patients", type=int, default=None, help="Custom patient count override")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random PRNG seed")
    parser.add_argument("--synthetic-dir", type=str, default="data/synthetic", help="Output directory for synthetic data")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Output directory for validated processed data")
    args = parser.parse_args()

    # Configure generation
    if args.patients is not None:
        config = GeneratorConfig(num_patients=args.patients, seed=args.seed)
    else:
        config = GeneratorConfig.from_preset(args.preset, seed=args.seed)

    print("\n" + "=" * 78)
    print("  HEALTHCARE DATA GENERATION & VALIDATION PIPELINE")
    print("=" * 78)
    print(f"[*] Preset Scale     : {args.preset.upper()} ({config.num_patients} patients)")
    print(f"[*] Random PRNG Seed : {config.seed}")
    print(f"[*] Synthetic Target : {args.synthetic_dir}")
    print(f"[*] Processed Target : {args.processed_dir}")

    synthetic_path = PROJECT_ROOT / args.synthetic_dir
    processed_path = PROJECT_ROOT / args.processed_dir

    # Step 1: Generate synthetic records
    print("\n[1] Generating deterministic synthetic clinical records...")
    generator = ConfigurableHealthcareGenerator(config)
    raw_cohort = generator.generate_cohort()
    generator.save_to_disk(raw_cohort, synthetic_path)

    # Step 2: Validate, deduplicate, and attach provenance stamps
    print("\n[2] Validating records and stamping cryptographic provenance...")
    sanitizer = IngestionSanitizer()
    clean_cohort, audit_manifest = sanitizer.process_and_validate_cohort(raw_cohort, seed=config.seed)
    generator.save_to_disk(clean_cohort, processed_path)

    # Save audit manifest
    manifest_file = processed_path / "audit_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(audit_manifest, f, indent=2)
    print(f"[+] Saved audit manifest -> {manifest_file.name}")

    print("\n[V] DATA GENERATION & SANITIZATION SUMMARY:")
    print("-" * 50)
    for entity, count in audit_manifest["counts_sanitized"].items():
        print(f"    - {entity:<20}: {count:>6} valid records")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
