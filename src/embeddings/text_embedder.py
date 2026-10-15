"""
Text Embedding Pipeline using Sentence Transformers
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements dense text embedding generation for clinical narrative passages using
Sentence Transformers (initially sentence-transformers/all-MiniLM-L6-v2, 384-d).
Operates strictly locally: zero PHI or healthcare text is ever sent to external APIs.
"""

import math
import hashlib
import logging
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("embeddings.text")

class DimensionMismatchError(Exception):
    """Raised when generated embedding vector dimension violates schema constraints."""
    pass

class TextEmbedder:
    """
    Generates dense semantic vector embeddings for clinical text passages.
    Configurable for standard Sentence Transformers architectures.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        device: str = "auto",
        use_fallback: bool = False
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.model = None
        self.use_fallback = use_fallback
        self.device = self._resolve_device(device)

        if not use_fallback:
            self._initialize_model()

    def _resolve_device(self, device: str) -> str:
        """Determines computation device (cuda vs cpu)."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _initialize_model(self) -> None:
        """Attempts to load the local SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device '{self.device}'...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Verify model output dimension matches configured dimension
            test_dim = self.model.get_sentence_embedding_dimension()
            if test_dim != self.dimension:
                raise DimensionMismatchError(
                    f"Model '{self.model_name}' produces {test_dim}-d vectors, but expected {self.dimension}-d."
                )
            logger.info(f"[V] SentenceTransformer '{self.model_name}' loaded successfully ({self.dimension} dimensions).")
        except Exception as e:
            logger.warning(f"[!] Could not load SentenceTransformer '{self.model_name}': {e}")
            logger.warning("    Activating deterministic local feature encoder for testing and offline execution.")
            logger.warning("    To use official Sentence Transformers weights, run:")
            logger.warning("    pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cpu")
            self.use_fallback = True

    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """L2-normalizes vector to unit length (||v||_2 = 1.0) for exact cosine distance."""
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    def _deterministic_embed(self, text: str) -> List[float]:
        """
        Generates deterministic pseudo-semantic 384-d unit vector based on token hashes.
        Guarantees reproducible cosine distance behavior for offline tests, benchmarks, and validation.
        """
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            for d in range(8):
                dim_idx = (h + d * 31) % self.dimension
                val = (((h >> (d * 4)) & 0xFF) / 128.0) - 1.0
                vec[dim_idx] += val

        return self.normalize_vector(vec)

    def validate_dimension(self, vector: List[float]) -> None:
        """Validates that a vector matches the expected dimension."""
        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single clinical text passage into a normalized vector."""
        if self.use_fallback or self.model is None:
            vec = self._deterministic_embed(text)
        else:
            raw_vec = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            vec = raw_vec.tolist()

        self.validate_dimension(vec)
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encodes a batch of clinical text passages."""
        if not texts:
            return []

        if self.use_fallback or self.model is None:
            vectors = [self._deterministic_embed(t) for t in texts]
        else:
            raw_vecs = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            vectors = [v.tolist() for v in raw_vecs]

        for v in vectors:
            self.validate_dimension(v)

        return vectors
