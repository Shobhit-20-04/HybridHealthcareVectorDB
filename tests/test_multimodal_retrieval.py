"""
Unit & Integration Tests for Multimodal Storage & Retrieval Pipeline
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Verifies:
1. Deterministic synthetic phantom PNG generation and PNG header structure
2. Cryptographic SHA-256 asset integrity verification and file existence audit
3. Modality-specific Image-to-Image retrieval (cosine distance, ranking, self-identity)
4. Cross-Modal Text-to-Image retrieval (CLIP text-tower to vision vectors)
5. Strict embedding space invariance (rejection of incompatible 384-d vectors)
6. Relational metadata predicate filtering and empty result handling
7. Input validation and edge-case handling
"""

import os
import hashlib
import tempfile
from pathlib import Path
import pytest

from database.connection import get_connection
from src.embeddings.image_store import (
    generate_synthetic_phantom_png,
    ImageAssetManager,
    IncompatibleEmbeddingSpaceError
)
from src.embeddings.image_embedder import ImageEmbedder
from src.retrieval.multimodal_search import MultimodalSearchEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fixture to provide an initialized search engine
@pytest.fixture(scope="module")
def multimodal_engine():
    embedder = ImageEmbedder(use_fallback=True)
    return MultimodalSearchEngine(embedder=embedder)

@pytest.fixture(scope="module")
def asset_manager():
    embedder = ImageEmbedder(use_fallback=True)
    return ImageAssetManager(embedder=embedder)

# =============================================================================
# 1. Synthetic Phantom Asset Generation Tests
# =============================================================================

def test_generate_synthetic_phantom_png():
    """Validates pure-Python synthetic phantom generator produces valid PNG files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test_phantom.png"
        path, sha256_hash, file_size = generate_synthetic_phantom_png(
            output_path=out_file,
            modality="DX",
            body_site="Chest",
            image_id="00000000-0000-0000-0000-000000000001",
            width=64,
            height=64
        )

        assert path.exists()
        assert file_size > 0
        assert len(sha256_hash) == 64

        # Verify PNG 8-byte magic header
        with open(path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n", "Generated file is not a valid PNG."

            # Verify cryptographic SHA-256 matches actual file bytes
            f.seek(0)
            actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == sha256_hash

def test_synthetic_phantom_determinism():
    """Ensures identical image IDs yield identical byte sequences and checksums."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p1 = Path(tmp_dir) / "phantom_1.png"
        p2 = Path(tmp_dir) / "phantom_2.png"

        _, hash1, size1 = generate_synthetic_phantom_png(p1, "CT", "Brain", "fixed-uuid-1234")
        _, hash2, size2 = generate_synthetic_phantom_png(p2, "CT", "Brain", "fixed-uuid-1234")

        assert hash1 == hash2
        assert size1 == size2
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            assert f1.read() == f2.read()

# =============================================================================
# 2. Database Asset Integrity & Verification Audit Tests
# =============================================================================

def test_asset_integrity_audit(asset_manager):
    """Audits database records against physical files on disk."""
    audit_res = asset_manager.verify_asset_integrity()
    assert audit_res["total_records"] > 0
    assert audit_res["verified_count"] == audit_res["total_records"]
    assert audit_res["missing_count"] == 0
    assert audit_res["corrupted_count"] == 0
    assert audit_res["audit_passed"] is True

def test_database_table_counts():
    """Verifies image_embeddings has been populated for all medical_images."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM medical_images;")
            n_images = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM image_embeddings WHERE model_id = 'openai/clip-vit-base-patch32';")
            n_embeddings = cur.fetchone()[0]

    assert n_images > 0
    assert n_embeddings == n_images

# =============================================================================
# 3. Image-to-Image Retrieval Tests
# =============================================================================

def test_image_to_image_self_retrieval(multimodal_engine):
    """
    Given an existing image asset, an Image-to-Image similarity search must
    retrieve the query image at Rank #1 with Cosine Distance ~ 0.0 (Similarity ~ 1.0).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT synthetic_file_path, synthetic_image_num 
                FROM medical_images 
                WHERE is_verified_asset = TRUE 
                LIMIT 1;
            """)
            sample = cur.fetchone()

    assert sample is not None
    rel_path, expected_img_num = sample
    abs_path = PROJECT_ROOT / rel_path

    res = multimodal_engine.search_image_to_image(query_image=abs_path, top_k=5)
    assert res["search_type"] == "image_to_image"
    assert res["returned_count"] > 0

    top_hit = res["results"][0]
    assert top_hit["rank"] == 1
    assert top_hit["synthetic_image_num"] == expected_img_num
    assert pytest.approx(top_hit["cosine_distance"], abs=1e-4) == 0.0
    assert pytest.approx(top_hit["cosine_similarity"], abs=1e-4) == 1.0

def test_image_to_image_monotonicity(multimodal_engine):
    """Verifies result distances are sorted in strictly ascending order."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT synthetic_file_path FROM medical_images LIMIT 1;")
            rel_path = cur.fetchone()[0]

    res = multimodal_engine.search_image_to_image(query_image=PROJECT_ROOT / rel_path, top_k=5)
    distances = [r["cosine_distance"] for r in res["results"]]
    for i in range(len(distances) - 1):
        assert distances[i] <= distances[i + 1], "Results are not monotonically sorted by distance."

# =============================================================================
# 4. Cross-Modal Text-to-Image Retrieval Tests
# =============================================================================

def test_cross_modal_text_to_image_retrieval(multimodal_engine):
    """Executes a natural language clinical query to find relevant imaging studies."""
    query = "bilateral lung opacity pulmonary infiltrates chest radiograph"
    res = multimodal_engine.search_text_to_image(query_text=query, top_k=5)

    assert res["search_type"] == "cross_modal_text_to_image"
    assert res["top_k"] == 5
    assert res["returned_count"] <= 5
    assert len(res["results"]) > 0

    first = res["results"][0]
    assert "synthetic_image_num" in first
    assert "modality" in first
    assert "body_site" in first
    assert "patient_age" in first
    assert "cosine_distance" in first
    assert "cosine_similarity" in first
    assert 0.0 <= first["cosine_distance"] <= 2.0
    assert -1.0 <= first["cosine_similarity"] <= 1.0

# =============================================================================
# 5. Strict Embedding Space Protection Tests
# =============================================================================

def test_incompatible_embedding_space_rejection(multimodal_engine):
    """
    Passing a 384-dimensional vector (or any dimension != 512) into multimodal
    engine MUST raise IncompatibleEmbeddingSpaceError.
    """
    incompatible_384d_vector = [0.05] * 384
    with pytest.raises(IncompatibleEmbeddingSpaceError) as exc_info:
        multimodal_engine.search_image_to_image(query_image=incompatible_384d_vector)

    assert "cannot compare 384-dimensional vector" in str(exc_info.value)
    assert "512-dimensional multimodal image space" in str(exc_info.value)

def test_incompatible_embedding_space_empty_or_bad_vector(multimodal_engine):
    """Validates that non-numeric or malformed vectors are rejected."""
    with pytest.raises(IncompatibleEmbeddingSpaceError):
        multimodal_engine.validate_vector_space([0.1, 0.2])  # 2-d vector

# =============================================================================
# 6. Relational Metadata Filtering Tests
# =============================================================================

def test_modality_filtering(multimodal_engine):
    """Verifies that results only contain the requested modality."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT synthetic_file_path FROM medical_images WHERE modality = 'DX' LIMIT 1;")
            rel_path = cur.fetchone()[0]

    res = multimodal_engine.search_image_to_image(
        query_image=PROJECT_ROOT / rel_path,
        modality="DX",
        top_k=5
    )

    assert res["returned_count"] > 0
    for r in res["results"]:
        assert r["modality"] == "DX"

def test_empty_results_on_restrictive_filter(multimodal_engine):
    """Verifies that an unmatchable metadata filter returns 0 results gracefully."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT synthetic_file_path FROM medical_images LIMIT 1;")
            rel_path = cur.fetchone()[0]

    res = multimodal_engine.search_image_to_image(
        query_image=PROJECT_ROOT / rel_path,
        modality="NON_EXISTENT_MODALITY",
        top_k=5
    )

    assert res["returned_count"] == 0
    assert res["results"] == []

# =============================================================================
# 7. Input Validation & Error Handling
# =============================================================================

def test_missing_image_file_raises_error(multimodal_engine):
    """Searching for a non-existent image file raises FileNotFoundError."""
    non_existent = PROJECT_ROOT / "data" / "synthetic" / "images" / "ghost_file.png"
    with pytest.raises(FileNotFoundError):
        multimodal_engine.search_image_to_image(query_image=non_existent)

def test_empty_text_query_raises_error(multimodal_engine):
    """Empty or whitespace-only clinical query text raises ValueError."""
    with pytest.raises(ValueError) as exc:
        multimodal_engine.search_text_to_image(query_text="   ")
    assert "cannot be empty" in str(exc.value)

def test_invalid_top_k_raises_error(multimodal_engine):
    """Non-positive top_k raises ValueError."""
    with pytest.raises(ValueError):
        multimodal_engine.search_text_to_image(query_text="valid query", top_k=0)
    with pytest.raises(ValueError):
        multimodal_engine.search_text_to_image(query_text="valid query", top_k=-3)
