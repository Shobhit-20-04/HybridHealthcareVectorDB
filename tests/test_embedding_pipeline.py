"""
Unit and Integration Tests for Document Preprocessing, Chunking, and Vector Storage
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval
"""

import pytest
import uuid
import math
from src.embeddings.chunker import clean_clinical_text, ClinicalDocumentChunker
from src.embeddings.text_embedder import TextEmbedder, DimensionMismatchError
from src.embeddings.vector_store import PgVectorStore
from database.connection import get_connection

def test_clean_clinical_text():
    """Verify that clinical text is properly normalized while preserving structure."""
    raw = "  CHIEF COMPLAINT:\tSevere dyspnea. \n\n\n\n  HISTORY:   Patient presented with edema.  \xa0 "
    cleaned = clean_clinical_text(raw)
    assert "\t" not in cleaned
    assert "\xa0" not in cleaned
    assert "  " not in cleaned
    assert "CHIEF COMPLAINT: Severe dyspnea." in cleaned
    assert "HISTORY: Patient presented with edema." in cleaned

def test_document_chunker_sliding_window():
    """Verify sliding window chunking, overlap, contiguous indices, and metadata retention."""
    chunker = ClinicalDocumentChunker(chunk_size_words=20, chunk_overlap_words=5)
    doc_id = str(uuid.uuid4())
    text = " ".join([f"word{i}" for i in range(50)])  # 50 words -> should create 3 chunks

    chunks = chunker.chunk_document(doc_id, text, doc_metadata={"dept": "Cardiology"})
    assert len(chunks) == 3

    for idx, c in enumerate(chunks):
        assert c["chunk_index"] == idx
        assert c["document_id"] == doc_id
        assert c["char_length"] == len(c["chunk_text"])
        assert c["token_count"] > 0
        assert c["doc_metadata"]["dept"] == "Cardiology"
        # Deterministic UUIDv5 check
        expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{idx}"))
        assert c["chunk_id"] == expected_uuid

def test_document_chunker_short_and_empty():
    """Verify edge cases: short text and empty text."""
    chunker = ClinicalDocumentChunker(chunk_size_words=50)
    doc_id = str(uuid.uuid4())

    # Empty text
    assert chunker.chunk_document(doc_id, "") == []
    assert chunker.chunk_document(doc_id, "   \n\t  ") == []

    # Short text
    chunks = chunker.chunk_document(doc_id, "Stable outpatient follow-up.")
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["chunk_text"] == "Stable outpatient follow-up."

def test_text_embedder_normalization_and_dimension():
    """Verify that vector output has exact dimension and unit Euclidean norm."""
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    vec = embedder.embed_text("Patient admitted for cardiac catheterization.")

    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)

    # Batch test
    batch_vecs = embedder.embed_batch(["Note 1", "Note 2", "Note 3"])
    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert len(v) == 384
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-4)

def test_dimension_mismatch_raises_error():
    """Verify that dimension mismatches raise DimensionMismatchError."""
    embedder = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    bad_vector = [0.1] * 512

    with pytest.raises(DimensionMismatchError, match="Vector dimension mismatch"):
        embedder.validate_dimension(bad_vector)

def test_pgvector_column_dimension_verification():
    """Verify that PgVectorStore accurately verifies the database column dimension."""
    store = PgVectorStore(dimension=384)
    col_dim = store.verify_column_dimension("chunk_embeddings", "embedding")
    assert col_dim == 384

    # Mismatched expected dimension should raise error
    bad_store = PgVectorStore(dimension=768)
    with pytest.raises(DimensionMismatchError, match="expects vector\\(384\\)"):
        bad_store.verify_column_dimension("chunk_embeddings", "embedding")

def test_idempotent_embedding_storage():
    """Verify that running vector storage repeatedly does not create duplicate rows."""
    store = PgVectorStore(model_id="sentence-transformers/all-MiniLM-L6-v2", dimension=384)

    # First run
    res1 = store.ingest_chunk_embeddings(batch_size=32)
    counts1 = store.get_embedding_counts()

    # Second run should identify 0 un-embedded chunks and maintain identical row counts
    res2 = store.ingest_chunk_embeddings(batch_size=32)
    counts2 = store.get_embedding_counts()

    assert counts1["model_embeddings"] == counts2["model_embeddings"]
    assert counts2["unembedded_chunks"] == 0
    assert counts2["orphan_embeddings"] == 0

def test_pgvector_cosine_distance_query():
    """Verify that pgvector cosine distance operator <=> performs valid nearest-neighbor retrieval."""
    store = PgVectorStore(model_id="sentence-transformers/all-MiniLM-L6-v2", dimension=384)
    query_vec = store.embedder.embed_text("shortness of breath and congestive heart failure")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ce.chunk_id, (ce.embedding <=> %s::vector) AS dist
                FROM chunk_embeddings ce
                WHERE ce.model_id = %s
                ORDER BY ce.embedding <=> %s::vector
                LIMIT 5;
            """, (query_vec, store.model_id, query_vec))
            results = cur.fetchall()

    assert len(results) > 0
    # Distances must be sorted ascending
    distances = [r[1] for r in results]
    assert distances == sorted(distances)
    for d in distances:
        assert 0.0 <= d <= 2.0  # Cosine distance is bounded [0, 2]
