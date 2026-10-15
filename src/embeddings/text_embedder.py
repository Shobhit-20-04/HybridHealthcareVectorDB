"""
Text Embedding Pipeline
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements dense text embedding generation using Sentence Transformers.
Includes deterministic fallback for offline execution, unit testing, and isolated benchmarking.
"""

import math
import hashlib
from typing import List, Optional

class TextEmbedder:
    """Generates 384-dimensional dense semantic vectors for clinical text."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: str = "cpu",
        use_fallback: bool = False
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.model = None
        self.use_fallback = use_fallback

        if not use_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name, device=device)
            except Exception as e:
                # Fall back gracefully to deterministic encoder if offline/missing weights
                print(f"[!] Warning: Could not initialize SentenceTransformer '{model_name}': {e}")
                print("    Activating deterministic pseudo-semantic fallback encoder for testing/benchmarks.")
                self.use_fallback = True

    def _normalize(self, vector: List[float]) -> List[float]:
        """L2-normalizes vector to unit length for exact cosine similarity."""
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    def _deterministic_embed(self, text: str) -> List[float]:
        """
        Generates deterministic pseudo-semantic 384-d unit vector based on n-gram hashes.
        Ensures consistent cosine distance behavior for offline tests and database validation.
        """
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for w_idx, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            for d in range(8):
                dim_idx = (h + d * 31) % self.dimension
                val = (((h >> (d * 4)) & 0xFF) / 128.0) - 1.0
                vec[dim_idx] += val

        return self._normalize(vec)

    def embed_text(self, text: str) -> List[float]:
        """Encodes single clinical text passage into a normalized vector."""
        if self.use_fallback or self.model is None:
            return self._deterministic_embed(text)
        
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encodes batch of clinical text passages."""
        if self.use_fallback or self.model is None:
            return [self._deterministic_embed(t) for t in texts]

        vectors = self.model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return [v.tolist() for v in vectors]
