"""
Clinical Retrieval-Augmented Generation (RAG) Module
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Provides:
- Configurable LLM Providers (Retrieval-Only, Deterministic Mock, Local Ollama, Opt-In External API)
- Separate Quality and Groundedness Evaluation Framework
- Headless Clinical RAG Pipeline
"""

from src.rag.llm_provider import (
    BaseLLMProvider,
    RetrievalOnlyProvider,
    DeterministicMockLLMProvider,
    LocalOllamaProvider,
    ExternalAPIProvider,
    ExternalAccessBlockedError,
    LLMResponse,
    get_llm_provider
)
from src.rag.evaluator import (
    RetrievalQualityEvaluator,
    GenerationGroundednessEvaluator,
    RAGEvaluator,
    RetrievalQualityMetrics,
    GroundednessMetrics
)
from src.rag.pipeline import ClinicalRAGPipeline, ClinicalRAGResult

__all__ = [
    "BaseLLMProvider",
    "RetrievalOnlyProvider",
    "DeterministicMockLLMProvider",
    "LocalOllamaProvider",
    "ExternalAPIProvider",
    "ExternalAccessBlockedError",
    "LLMResponse",
    "get_llm_provider",
    "RetrievalQualityEvaluator",
    "GenerationGroundednessEvaluator",
    "RAGEvaluator",
    "RetrievalQualityMetrics",
    "GroundednessMetrics",
    "ClinicalRAGPipeline",
    "ClinicalRAGResult",
]
