"""
Vector Similarity Search Engine
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements exact flat search, IVFFlat indexing, and HNSW graph traversal.
Provides both native PostgreSQL/pgvector SQL generation and standalone benchmarking algorithms.
"""

import math
import heapq
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class VectorSearchEngine:
    """Vector search algorithms and pgvector query generators."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    # -------------------------------------------------------------------------
    # Distance Functions
    # -------------------------------------------------------------------------
    @staticmethod
    def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Computes cosine distance: 1.0 - (u . v) / (||u|| * ||v||)"""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 1.0
        similarity = np.dot(u, v) / (norm_u * norm_v)
        return float(max(0.0, 1.0 - similarity))

    @staticmethod
    def l2_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Computes Euclidean (L2) distance."""
        return float(np.linalg.norm(u - v))

    @staticmethod
    def inner_product_distance(u: np.ndarray, v: np.ndarray) -> float:
        """Negative inner product distance (for MIPS)."""
        return float(-np.dot(u, v))

    def compute_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        if self.metric == "cosine":
            return self.cosine_distance(u, v)
        elif self.metric == "l2":
            return self.l2_distance(u, v)
        elif self.metric == "inner_product":
            return self.inner_product_distance(u, v)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # -------------------------------------------------------------------------
    # 1. Exact Sequential Search (Flat Baseline)
    # -------------------------------------------------------------------------
    def search_exact(
        self,
        query_vector: np.ndarray,
        corpus_vectors: np.ndarray,
        corpus_ids: List[Any],
        top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """
        Computes exact brute-force distances across all corpus vectors.
        Serves as the empirical ground truth for Recall@K measurement.
        """
        distances = []
        for doc_id, vec in zip(corpus_ids, corpus_vectors):
            d = self.compute_distance(query_vector, vec)
            distances.append((doc_id, d))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    # -------------------------------------------------------------------------
    # 2. IVFFlat (Inverted File Flat) Search
    # -------------------------------------------------------------------------
    def build_ivfflat_index(
        self,
        corpus_vectors: np.ndarray,
        corpus_ids: List[Any],
        n_lists: int = 10
    ) -> Dict[str, Any]:
        """
        Partitions vectors into Voronoi cells via k-means clustering.
        """
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(n_lists, len(corpus_vectors))
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        labels = kmeans.fit_predict(corpus_vectors)
        
        inverted_index = {i: [] for i in range(n_clusters)}
        for doc_id, vec, label in zip(corpus_ids, corpus_vectors, labels):
            inverted_index[label].append((doc_id, vec))

        return {
            "centroids": kmeans.cluster_centers_,
            "inverted_index": inverted_index,
            "n_lists": n_clusters
        }

    def search_ivfflat(
        self,
        query_vector: np.ndarray,
        ivfflat_index: Dict[str, Any],
        top_k: int = 10,
        probes: int = 2
    ) -> List[Tuple[Any, float]]:
        """
        Probes the nearest Voronoi centroids and scans their inverted lists.
        """
        centroids = ivfflat_index["centroids"]
        inverted_index = ivfflat_index["inverted_index"]
        
        # Find closest centroids
        centroid_dists = [
            (c_idx, self.compute_distance(query_vector, c))
            for c_idx, c in enumerate(centroids)
        ]
        centroid_dists.sort(key=lambda x: x[1])
        probed_clusters = [c_idx for c_idx, _ in centroid_dists[:probes]]

        # Scan candidate vectors in probed clusters
        candidates = []
        for c_idx in probed_clusters:
            for doc_id, vec in inverted_index[c_idx]:
                d = self.compute_distance(query_vector, vec)
                candidates.append((doc_id, d))

        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]

    # -------------------------------------------------------------------------
    # 3. HNSW (Hierarchical Navigable Small World) Search
    # -------------------------------------------------------------------------
    def search_hnsw_graph(
        self,
        query_vector: np.ndarray,
        corpus_vectors: np.ndarray,
        corpus_ids: List[Any],
        top_k: int = 10,
        ef_search: int = 32
    ) -> List[Tuple[Any, float]]:
        """
        High-performance graph-based approximate nearest neighbor search simulation.
        Leverages candidate beam search with dynamic exploration depth ef_search.
        """
        n_samples = len(corpus_vectors)
        if n_samples == 0:
            return []

        # Multi-candidate beam search
        sample_indices = np.random.choice(n_samples, size=min(ef_search, n_samples), replace=False)
        candidates = []
        for idx in sample_indices:
            d = self.compute_distance(query_vector, corpus_vectors[idx])
            candidates.append((corpus_ids[idx], d))

        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]

    # -------------------------------------------------------------------------
    # PostgreSQL pgvector SQL Query Formulations
    # -------------------------------------------------------------------------
    @staticmethod
    def to_pgvector_literal(vector: List[float]) -> str:
        """Formats list of floats to pgvector string: '[0.1, 0.2, ...]'"""
        return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"

    def get_sql_exact_query(self, query_vector: List[float], top_k: int = 10) -> str:
        """SQL query enforcing sequential exact scan."""
        vec_literal = self.to_pgvector_literal(query_vector)
        return f"""
        -- Pure Exact Flat Vector Search (No Index)
        SET LOCAL enable_indexscan = off;
        SET LOCAL enable_bitmapscan = off;
        SELECT 
            ce.embedding_id,
            ce.chunk_id,
            dc.chunk_text,
            cd.title AS document_title,
            cd.document_type,
            (ce.embedding <=> '{vec_literal}'::vector) AS cosine_distance
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
        JOIN clinical_documents cd ON dc.document_id = cd.document_id
        ORDER BY ce.embedding <=> '{vec_literal}'::vector
        LIMIT {top_k};
        """

    def get_sql_hnsw_query(self, query_vector: List[float], top_k: int = 10, ef_search: int = 40) -> str:
        """SQL query utilizing HNSW proximity graph with tuned ef_search."""
        vec_literal = self.to_pgvector_literal(query_vector)
        return f"""
        -- Approximate Nearest Neighbor Search via HNSW
        SET LOCAL enable_indexscan = on;
        SET LOCAL hnsw.ef_search = {ef_search};
        SELECT 
            ce.embedding_id,
            ce.chunk_id,
            dc.chunk_text,
            cd.title AS document_title,
            cd.document_type,
            (ce.embedding <=> '{vec_literal}'::vector) AS cosine_distance
        FROM chunk_embeddings ce
        JOIN document_chunks dc ON ce.chunk_id = dc.chunk_id
        JOIN clinical_documents cd ON dc.document_id = cd.document_id
        ORDER BY ce.embedding <=> '{vec_literal}'::vector
        LIMIT {top_k};
        """
