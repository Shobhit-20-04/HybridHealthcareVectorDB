"""
Configurable LLM Provider Interface & Implementations for Clinical RAG
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Implements:
1. BaseLLMProvider (Abstract Base Class)
2. RetrievalOnlyProvider (Default offline baseline when no LLM is configured)
3. DeterministicMockLLMProvider (Deterministic offline generator for testing/local evaluation)
4. LocalOllamaProvider (Local LLM execution without external network transmission)
5. ExternalAPIProvider (Strictly opt-in external provider with data privacy safeguards)
6. ExternalAccessBlockedError (Raised when external transmission is attempted without opt-in)
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.llm_provider")

REFUSAL_PHRASE = "The available documents do not establish an answer to the question."

class ExternalAccessBlockedError(Exception):
    """Raised when an external LLM call is attempted without explicit opt-in confirmation."""
    pass

@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""
    content: str
    provider_name: str
    model_name: str
    citations: List[str] = field(default_factory=list)
    has_evidence: bool = True
    abstention_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers in the Clinical RAG system."""

    def __init__(self, model_name: str, is_external: bool = False, allow_external: bool = False):
        self.model_name = model_name
        self.is_external = is_external
        self.allow_external = allow_external

        # Strict security guard against unintentional data exfiltration
        if self.is_external and not self.allow_external:
            # Check environment fallback
            env_opt_in = os.getenv("ALLOW_EXTERNAL_LLM", "false").strip().lower() in ("true", "1", "yes")
            if env_opt_in:
                self.allow_external = True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generates an evidence-grounded answer based strictly on retrieved passages.
        Must cite retrieved document IDs and explicitly refuse if evidence is inadequate.
        """
        pass

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extracts document identifiers cited in square brackets (e.g. [SYN-DOC-0042-01-01], [DOC-0042], or [doc-uuid])."""
        pattern = r"\[((?:SYN-DOC|DOC)-[A-Za-z0-9\-]+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for m in matches:
            norm = m.upper() if (m.upper().startswith("DOC-") or m.upper().startswith("SYN-DOC-")) else m.lower()
            if norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        return deduped

class RetrievalOnlyProvider(BaseLLMProvider):
    """
    Retrieval-only provider that performs no language model generation.
    Displays retrieved document passages and citations without synthesizing text.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="none (retrieval-only)", is_external=False, allow_external=False)

    def generate(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        doc_ids = []
        for p in passages:
            doc_id = p.get("synthetic_document_num") or p.get("document_id")
            if doc_id:
                doc_ids.append(str(doc_id))

        content = (
            "[RETRIEVAL-ONLY MODE] Generation skipped. "
            f"Retrieved {len(passages)} relevant healthcare passages from the database. "
            "Examine the displayed source snippets above for direct clinical review."
        )

        return LLMResponse(
            content=content,
            provider_name="retrieval_only",
            model_name=self.model_name,
            citations=doc_ids,
            has_evidence=bool(passages),
            abstention_triggered=False,
            metadata={"retrieved_count": len(passages)}
        )

class DeterministicMockLLMProvider(BaseLLMProvider):
    """
    Offline, deterministic rule-based LLM provider.
    Runs locally with zero external network traffic, ensuring complete PHI privacy.
    Synthesizes grounded answers strictly from passage text, separates evidence from uncertainty,
    and explicitly triggers abstention when evidence is insufficient or missing.
    """

    def __init__(
        self,
        model_name: str = "deterministic-clinical-evaluator-v1",
        relevance_threshold: float = 0.82
    ):
        super().__init__(model_name=model_name, is_external=False, allow_external=False)
        self.relevance_threshold = relevance_threshold

    def generate(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        # 1. Check for empty passages
        if not passages:
            content = (
                f"{REFUSAL_PHRASE}\n\n"
                "**Clinical Uncertainty & Limitations:**\n"
                "- Zero qualifying clinical documents were retrieved from the database matching this query."
            )
            return LLMResponse(
                content=content,
                provider_name="deterministic_mock",
                model_name=self.model_name,
                citations=[],
                has_evidence=False,
                abstention_triggered=True,
                metadata={"reason": "empty_passages"}
            )

        # 2. Extract query keywords from prompt
        # Prompt typically has: "Question: <question>"
        question_match = re.search(r"Question:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        question_text = question_match.group(1).strip() if question_match else prompt
        
        # Stop words filter for basic clinical keywords
        stop_words = {
            "what", "is", "the", "are", "and", "or", "in", "of", "to", "for", "with",
            "a", "an", "on", "at", "by", "from", "how", "does", "patient", "clinical",
            "medical", "show", "tell", "me", "about", "find", "were", "was"
        }
        query_words = set(
            re.findall(r"\b[a-zA-Z]{3,}\b", question_text.lower())
        ) - stop_words

        # 3. Check relevance of top passages
        # Check cosine distance: distances > relevance_threshold indicate weak semantic proximity
        best_distance = passages[0].get("cosine_distance", 1.0)
        
        # Check lexical keyword overlap across all passages
        matched_passages = []
        for p in passages:
            snippet = p.get("snippet", "").lower()
            overlap = [w for w in query_words if w in snippet]
            if overlap or (p.get("cosine_distance", 1.0) < 0.45):
                matched_passages.append((p, overlap))

        has_any_overlap = any(bool(m[1]) for m in matched_passages)
        # 4. Abstention trigger: If no passages matched, or no query keywords appear in passages and distance > 0.50
        if not matched_passages or (not has_any_overlap and best_distance > 0.50) or (best_distance >= self.relevance_threshold):
            content = (
                f"{REFUSAL_PHRASE}\n\n"
                "**Clinical Uncertainty & Limitations:**\n"
                f"- The retrieved database passages (best cosine distance: {best_distance:.4f}) do not contain "
                f"sufficient documented evidence regarding the query terms: {sorted(list(query_words))}.\n"
                "- No definitive clinical conclusion or treatment recommendation can be substantiated from available records."
            )
            return LLMResponse(
                content=content,
                provider_name="deterministic_mock",
                model_name=self.model_name,
                citations=[],
                has_evidence=False,
                abstention_triggered=True,
                metadata={"best_distance": best_distance, "query_words": list(query_words)}
            )

        # 5. Synthesize grounded answer from matched passages
        evidence_points = []
        cited_ids = []

        for p, overlap in matched_passages[:3]:
            doc_id = p.get("synthetic_document_num") or p.get("document_id", "DOC-UNKNOWN")
            dept = p.get("department", "General")
            snippet = p.get("snippet", "").strip()

            if doc_id not in cited_ids:
                cited_ids.append(doc_id)

            # Extract informative sentences containing overlap terms or clinical sections
            sentences = re.split(r"(?<=[.!?])\s+", snippet)
            salient_sents = [s for s in sentences if any(w in s.lower() for w in query_words)]
            if not salient_sents:
                salient_sents = sentences[:2]

            excerpt = " ".join(salient_sents[:2]).strip()
            evidence_points.append(f"- According to [{doc_id}] ({dept}): \"{excerpt}\"")

        evidence_section = "\n".join(evidence_points)
        uncertainty_section = (
            "**Clinical Uncertainty & Limitations:**\n"
            f"- Evidence is derived strictly from {len(cited_ids)} retrieved synthetic clinical document chunk(s).\n"
            "- Longitudinal disease progression, unmeasured confounders, and contraindications not recorded in "
            "these specific excerpts cannot be determined from the available records."
        )

        content = (
            "**Documented Evidence:**\n"
            f"{evidence_section}\n\n"
            f"{uncertainty_section}"
        )

        return LLMResponse(
            content=content,
            provider_name="deterministic_mock",
            model_name=self.model_name,
            citations=cited_ids,
            has_evidence=True,
            abstention_triggered=False,
            metadata={"matched_passages": len(matched_passages), "best_distance": best_distance}
        )

class LocalOllamaProvider(BaseLLMProvider):
    """
    Local Ollama provider executing on the host system (e.g. http://localhost:11434).
    Processes prompts locally without sending data to external cloud APIs.
    """

    def __init__(
        self,
        model_name: str = "llama3:latest",
        endpoint: str = "http://localhost:11434/api/generate",
        timeout_seconds: int = 45
    ):
        super().__init__(model_name=model_name, is_external=False, allow_external=False)
        self.endpoint = endpoint
        self.timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        if not passages:
            return LLMResponse(
                content=f"{REFUSAL_PHRASE}\n\nUncertainty: No context passages provided.",
                provider_name="local_ollama",
                model_name=self.model_name,
                citations=[],
                has_evidence=False,
                abstention_triggered=True
            )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": 0.1}
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                raw_text = data.get("response", "")
                citations = self.extract_citations(raw_text)
                abstention = REFUSAL_PHRASE.lower() in raw_text.lower()
                return LLMResponse(
                    content=raw_text,
                    provider_name="local_ollama",
                    model_name=self.model_name,
                    citations=citations,
                    has_evidence=not abstention,
                    abstention_triggered=abstention,
                    metadata={"eval_duration": data.get("eval_duration")}
                )
        except Exception as e:
            logger.warning(f"[!] Local Ollama call failed ({e}). Falling back to deterministic evaluator.")
            fallback = DeterministicMockLLMProvider()
            return fallback.generate(prompt, passages, system_prompt)

class ExternalAPIProvider(BaseLLMProvider):
    """
    Opt-in external LLM provider (e.g. Gemini / OpenAI / Anthropic compatible).
    ENFORCES STRICT OPT-IN: Blocks execution unless allow_external=True.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        allow_external: bool = False,
        api_endpoint: Optional[str] = None
    ):
        super().__init__(model_name=model_name, is_external=True, allow_external=allow_external)
        self.api_key = api_key or os.getenv("EXTERNAL_LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.api_endpoint = api_endpoint

    def generate(
        self,
        prompt: str,
        passages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        # Mandatory Privacy Guard
        if not self.allow_external:
            raise ExternalAccessBlockedError(
                "DATA PRIVACY GUARD: External LLM access is blocked by default to prevent transmitting "
                "clinical passages to external APIs. To opt in, set allow_external=True or use --allow-external."
            )

        if not self.api_key:
            raise ValueError(
                "External LLM API key not configured. Set EXTERNAL_LLM_API_KEY or GEMINI_API_KEY in environment."
            )

        if not passages:
            return LLMResponse(
                content=f"{REFUSAL_PHRASE}\n\nUncertainty: No context passages provided.",
                provider_name="external_api",
                model_name=self.model_name,
                citations=[],
                has_evidence=False,
                abstention_triggered=True
            )

        # Gemini REST implementation via urllib
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            f"?key={self.api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": f"{system_prompt or ''}\n\n{prompt}"}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
                citations = self.extract_citations(raw_text)
                abstention = REFUSAL_PHRASE.lower() in raw_text.lower()
                return LLMResponse(
                    content=raw_text,
                    provider_name="external_api",
                    model_name=self.model_name,
                    citations=citations,
                    has_evidence=not abstention,
                    abstention_triggered=abstention
                )
        except Exception as e:
            raise RuntimeError(f"External API generation failed: {e}")

def get_llm_provider(
    provider_type: str = "mock",
    allow_external: bool = False,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None
) -> BaseLLMProvider:
    """Factory creating configured LLM provider according to security parameters."""
    p_type = provider_type.strip().lower()

    if p_type in ("none", "retrieval", "retrieval_only", "retrieval-only"):
        return RetrievalOnlyProvider()

    if p_type in ("mock", "deterministic", "offline"):
        return DeterministicMockLLMProvider(model_name=model_name or "deterministic-clinical-evaluator-v1")

    if p_type in ("ollama", "local"):
        return LocalOllamaProvider(model_name=model_name or "llama3:latest", endpoint=endpoint or "http://localhost:11434/api/generate")

    if p_type in ("external", "gemini", "openai"):
        return ExternalAPIProvider(
            model_name=model_name or "gemini-1.5-flash",
            api_key=api_key,
            allow_external=allow_external,
            api_endpoint=endpoint
        )

    raise ValueError(f"Unknown LLM provider type '{provider_type}'. Choose from: 'mock', 'retrieval-only', 'ollama', 'external'.")
