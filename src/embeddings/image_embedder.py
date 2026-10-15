"""
Multimodal Image & Cross-Modal Embedding Pipeline
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Generates 512-dimensional multimodal vision and cross-modal query vectors using CLIP / BiomedCLIP.
Includes deterministic offline fallback for reproducible testing and benchmark environments.
"""

import math
import hashlib
from pathlib import Path
from typing import List, Union

class ImageEmbedder:
    """Generates 512-dimensional normalized vision and cross-modal embeddings."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        dimension: int = 512,
        device: str = "cpu",
        use_fallback: bool = False
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        self.model = None
        self.processor = None
        self.use_fallback = use_fallback

        if not use_fallback:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model = CLIPModel.from_pretrained(model_name).to(device)
            except Exception as e:
                print(f"[!] Warning: Could not initialize CLIP model '{model_name}': {e}")
                print("    Activating deterministic multimodal fallback encoder.")
                self.use_fallback = True

    def _normalize(self, vector: List[float]) -> List[float]:
        """L2-normalizes vector to unit length."""
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0.0:
            return vector
        return [x / norm for x in vector]

    def _deterministic_embed(self, key_str: str) -> List[float]:
        """Generates deterministic pseudo-vision 512-d unit vector."""
        vec = [0.0] * self.dimension
        tokens = key_str.lower().split()
        if not tokens:
            vec[0] = 1.0
            return vec

        for t in tokens:
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16)
            for d in range(8):
                dim_idx = (h + d * 43) % self.dimension
                val = (((h >> (d * 4)) & 0xFF) / 128.0) - 1.0
                vec[dim_idx] += val

        return self._normalize(vec)

    def embed_image_path(self, image_path: Union[str, Path]) -> List[float]:
        """Encodes an image file into a 512-d normalized vector."""
        if self.use_fallback or self.model is None:
            # Deterministically derive embedding from image metadata path string
            return self._deterministic_embed(str(image_path))

        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().tolist()

    def embed_query_text(self, text: str) -> List[float]:
        """
        Cross-modal retrieval: Encodes a text query into the multimodal 512-d vector space
        to perform similarity search over medical images.
        """
        if self.use_fallback or self.model is None:
            return self._deterministic_embed(text)

        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().tolist()
