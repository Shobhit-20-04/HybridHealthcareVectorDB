"""
Clinical Document Preprocessor & Chunker
Project: Hybrid Relational–Vector Database Architecture for Semantic Healthcare Information Retrieval

Provides robust clinical text preprocessing and configurable passage chunking with:
1. Whitespace and formatting normalization while preserving clinical section boundaries.
2. Configurable token/word sliding window chunking with semantic overlap.
3. Strict retention of parent document IDs, chunk indices, and metadata.
4. Idempotent PostgreSQL insertion into document_chunks.
"""

import re
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from database.connection import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("embeddings.chunker")

def clean_clinical_text(text: str) -> str:
    """
    Normalizes clinical narrative text:
    - Replaces tabs and non-breaking spaces with standard space.
    - Preserves clinical section headers (e.g. 'HISTORY OF PRESENT ILLNESS:').
    - Collapses consecutive whitespace characters while preserving paragraph breaks.
    - Strips leading and trailing whitespace.
    """
    if not text:
        return ""
    
    # Normalize unicode non-breaking spaces and tabs
    cleaned = text.replace("\xa0", " ").replace("\t", " ")
    # Normalize Windows CRLF to LF
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Replace 3 or more consecutive newlines with 2 newlines (preserve paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Collapse multiple spaces and horizontal whitespace within lines
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    # Strip whitespace on individual lines
    lines = [line.strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()

class ClinicalDocumentChunker:
    """Splits clinical documents into overlapping token/word passages."""

    def __init__(
        self,
        chunk_size_words: int = 80,
        chunk_overlap_words: int = 15,
        min_chunk_words: int = 10
    ):
        if chunk_size_words <= 0:
            raise ValueError(f"chunk_size_words must be positive, got {chunk_size_words}")
        if chunk_overlap_words < 0 or chunk_overlap_words >= chunk_size_words:
            raise ValueError(f"chunk_overlap_words must be between 0 and chunk_size_words - 1")
        
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.min_chunk_words = min_chunk_words

    def chunk_document(
        self,
        document_id: str,
        text: str,
        doc_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Preprocesses and chunks a single clinical document.
        Retains source document_id, creates contiguous chunk_index (0..N-1),
        computes token count, char length, and derives deterministic UUIDv5 chunk_id.
        """
        cleaned_text = clean_clinical_text(text)
        if not cleaned_text:
            return []

        words = cleaned_text.split()
        if not words:
            return []

        # If document is short enough to fit in a single chunk
        if len(words) <= self.chunk_size_words:
            chunk_str = " ".join(words)
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:0"))
            return [{
                "chunk_id": chunk_id,
                "document_id": str(document_id),
                "chunk_index": 0,
                "chunk_text": chunk_str,
                "token_count": max(1, int(len(words) * 1.3)),
                "char_length": len(chunk_str),
                "doc_metadata": doc_metadata or {}
            }]

        chunks = []
        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size_words - self.chunk_overlap_words)

        while start < len(words):
            end = min(start + self.chunk_size_words, len(words))
            chunk_words = words[start:end]
            chunk_str = " ".join(chunk_words)

            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}:{chunk_idx}"))
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": str(document_id),
                "chunk_index": chunk_idx,
                "chunk_text": chunk_str,
                "token_count": max(1, int(len(chunk_words) * 1.3)),
                "char_length": len(chunk_str),
                "doc_metadata": doc_metadata or {}
            })
            chunk_idx += 1

            if end >= len(words):
                break
            start += step

        return chunks

    def process_and_store_chunks(
        self,
        force_rechunk: bool = False
    ) -> Dict[str, int]:
        """
        Reads clinical documents from PostgreSQL and idempotently populates document_chunks.
        If force_rechunk is False, only processes documents that have no existing chunks.
        """
        with get_connection(autocommit=False) as conn:
            with conn.cursor() as cur:
                if force_rechunk:
                    cur.execute("""
                        SELECT document_id, full_text, metadata 
                        FROM clinical_documents 
                        ORDER BY created_at;
                    """)
                else:
                    cur.execute("""
                        SELECT cd.document_id, cd.full_text, cd.metadata
                        FROM clinical_documents cd
                        LEFT JOIN document_chunks dc ON cd.document_id = dc.document_id
                        WHERE dc.chunk_id IS NULL
                        GROUP BY cd.document_id, cd.full_text, cd.metadata, cd.created_at
                        ORDER BY cd.created_at;
                    """)
                
                documents = cur.fetchall()
                logger.info(f"Retrieved {len(documents)} clinical documents requiring chunking.")

                total_chunks_stored = 0
                all_chunks = []

                for doc_id, text, metadata in documents:
                    doc_chunks = self.chunk_document(str(doc_id), text, metadata)
                    all_chunks.extend(doc_chunks)

                if all_chunks:
                    logger.info(f"Upserting {len(all_chunks)} chunks into document_chunks...")
                    upsert_stmt = """
                        INSERT INTO document_chunks (
                            chunk_id, document_id, chunk_index, chunk_text,
                            token_count, char_length, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            token_count = EXCLUDED.token_count,
                            char_length = EXCLUDED.char_length,
                            created_at = CURRENT_TIMESTAMP;
                    """
                    chunk_data = [
                        (
                            c["chunk_id"], c["document_id"], c["chunk_index"],
                            c["chunk_text"], c["token_count"], c["char_length"]
                        )
                        for c in all_chunks
                    ]
                    cur.executemany(upsert_stmt, chunk_data)
                    total_chunks_stored = len(chunk_data)

            conn.commit()

        logger.info(f"[V] Chunking completed: {len(documents)} documents processed, {total_chunks_stored} chunks upserted.")
        return {
            "documents_processed": len(documents),
            "chunks_stored": total_chunks_stored
        }
