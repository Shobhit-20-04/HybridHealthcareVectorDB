"""
Unit tests for text and multimodal embedding pipelines
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import math
import pytest
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder

def test_text_embedder_dimension_and_normalization():
    embedder = TextEmbedder(dimension=384, use_fallback=True)
    text = "Patient presenting with acute heart failure exacerbation."
    vec = embedder.embed_text(text)
    
    assert len(vec) == 384, f"Expected vector dimension 384, got {len(vec)}"
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5, f"Vector is not unit normalized: {norm}"

def test_text_embedder_batch():
    embedder = TextEmbedder(dimension=384, use_fallback=True)
    texts = ["Note 1", "Note 2", "Note 3"]
    vectors = embedder.embed_batch(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384

def test_image_embedder_dimension_and_cross_modal():
    embedder = ImageEmbedder(dimension=512, use_fallback=True)
    img_path = "/images/cr/chest_xray_001.png"
    img_vec = embedder.embed_image_path(img_path)
    assert len(img_vec) == 512, f"Expected 512 dimensions for image, got {len(img_vec)}"

    query_text = "Bilateral interstitial pulmonary infiltrates"
    q_vec = embedder.embed_query_text(query_text)
    assert len(q_vec) == 512
    norm = math.sqrt(sum(x * x for x in q_vec))
    assert abs(norm - 1.0) < 1e-5
