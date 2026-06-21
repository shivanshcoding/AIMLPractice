"""
Contextual Compression.

Extracts only query-relevant sentences from each chunk.
"""

from __future__ import annotations

import re

import structlog

from src.core.interfaces import BaseCompressor, BaseLLM
from src.core.models import CompressedContext, RetrievalResult, Chunk
from src.chunking.base import BaseChunkerMixin

logger = structlog.get_logger(__name__)

EXTRACT_PROMPT = """Given the following context and question, extract ONLY the sentences
that are directly relevant to answering the question. Return the relevant sentences verbatim.
If no sentences are relevant, return "NOT_RELEVANT".

Question: {query}

Context:
{context}

Relevant sentences:"""


class ContextualCompressor(BaseCompressor):
    """Extracts query-relevant sentences from each chunk using LLM."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    @property
    def compressor_name(self) -> str:
        return "contextual_compression"

    def compress(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> CompressedContext:
        original_tokens = sum(d.chunk.token_count for d in documents)
        compressed_docs: list[RetrievalResult] = []

        for doc in documents:
            try:
                prompt = EXTRACT_PROMPT.format(query=query, context=doc.chunk.content)
                extracted = self._llm.generate(prompt, temperature=0.0, max_tokens=1024)
                extracted = extracted.strip()

                if extracted and extracted != "NOT_RELEVANT":
                    new_chunk = doc.chunk.model_copy()
                    new_chunk.content = extracted
                    new_chunk.token_count = BaseChunkerMixin.count_tokens(extracted)
                    new_doc = doc.model_copy()
                    new_doc.chunk = new_chunk
                    compressed_docs.append(new_doc)

            except Exception as e:
                logger.warning("contextual_compression_failed", error=str(e))
                compressed_docs.append(doc)

        compressed_tokens = sum(d.chunk.token_count for d in compressed_docs)

        return CompressedContext(
            documents=compressed_docs,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            compression_method=self.compressor_name,
        )
