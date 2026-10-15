"""
Vector Embedding Ingestion Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Processes synthetic document chunks and medical images, generates normalized dense vectors
(384-d for text, 512-d for multimodal), and exports or inserts embeddings into PostgreSQL/pgvector.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder

def generate_and_save_embeddings(
    data_dir: Path,
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    image_model_id: str = "openai/clip-vit-base-patch32",
    use_fallback: bool = False
) -> Dict[str, int]:
    """Generates embeddings for synthetic chunks and images, saving to JSON fixtures."""
    chunks_file = data_dir / "document_chunks.json"
    images_file = data_dir / "medical_images.json"

    if not chunks_file.exists():
        raise FileNotFoundError(f"Missing {chunks_file}. Run synthetic_generator.py first.")

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    images = []
    if images_file.exists():
        with open(images_file, "r", encoding="utf-8") as f:
            images = json.load(f)

    print(f"[+] Loaded {len(chunks)} text chunks and {len(images)} image records.")

    # 1. Generate text chunk embeddings (384-dim)
    print(f"[+] Generating 384-d text embeddings using model '{text_model_id}'...")
    text_embedder = TextEmbedder(model_name=text_model_id, dimension=384, use_fallback=use_fallback)
    
    texts = [c["chunk_text"] for c in chunks]
    vectors = text_embedder.embed_batch(texts)

    chunk_embeddings = []
    for chunk, vec in zip(chunks, vectors):
        chunk_embeddings.append({
            "embedding_id": str(uuid.uuid4()),
            "chunk_id": chunk["chunk_id"],
            "model_id": text_model_id,
            "dimension": 384,
            "embedding": vec,
            "created_at": chunk["created_at"]
        })

    out_chunk_emb = data_dir / "chunk_embeddings.json"
    with open(out_chunk_emb, "w", encoding="utf-8") as f:
        json.dump(chunk_embeddings, f, indent=2)
    print(f"[+] Saved {len(chunk_embeddings)} chunk embeddings to {out_chunk_emb.name}")

    # 2. Generate multimodal image embeddings (512-dim)
    print(f"[+] Generating 512-d image embeddings using model '{image_model_id}'...")
    image_embedder = ImageEmbedder(model_name=image_model_id, dimension=512, use_fallback=use_fallback)

    image_embeddings = []
    for img in images:
        vec = image_embedder.embed_image_path(img["synthetic_file_path"])
        image_embeddings.append({
            "embedding_id": str(uuid.uuid4()),
            "image_id": img["image_id"],
            "model_id": image_model_id,
            "dimension": 512,
            "embedding": vec,
            "created_at": img["created_at"]
        })

    out_img_emb = data_dir / "image_embeddings.json"
    with open(out_img_emb, "w", encoding="utf-8") as f:
        json.dump(image_embeddings, f, indent=2)
    print(f"[+] Saved {len(image_embeddings)} image embeddings to {out_img_emb.name}")

    return {
        "chunk_embeddings_count": len(chunk_embeddings),
        "image_embeddings_count": len(image_embeddings)
    }

if __name__ == "__main__":
    base_data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
    stats = generate_and_save_embeddings(base_data_dir)
    print(f"[V] Embedding generation completed: {stats}")
