"""
LLM-Based Context Compression.

Summarizes retrieved context to reduce token count before generation.
"""

from __future__ import annotations

import structlog

from src.core.interfaces import BaseCompressor, BaseLLM
from src.core.models import CompressedContext, RetrievalResult, Chunk
from src.chunking.base import BaseChunkerMixin

logger = structlog.get_logger(__name__)

SUMMARIZE_PROMPT = """Summarize the following context concisely while preserving all
key facts and information needed to answer the question. Be thorough but brief.

Question: {query}

Context:
{context}

Concise Summary:"""


class LLMCompressor(BaseCompressor):
    """LLM-based summarization compressor."""

    def __init__(self, llm: BaseLLM, target_ratio: float = 0.5) -> None:
        self._llm = llm
        self._target_ratio = target_ratio

    @property
    def compressor_name(self) -> str:
        return "llm_compression"

    def compress(
        self,
        query: str,
        documents: list[RetrievalResult],
    ) -> CompressedContext:
        original_tokens = sum(d.chunk.token_count for d in documents)

        # Combine all content
        combined = "\n\n".join(d.chunk.content for d in documents)

        try:
            prompt = SUMMARIZE_PROMPT.format(query=query, context=combined)
            target_tokens = int(original_tokens * self._target_ratio)
            summary = self._llm.generate(
                prompt, temperature=0.0, max_tokens=max(256, target_tokens)
            )
            summary = summary.strip()

            compressed_tokens = BaseChunkerMixin.count_tokens(summary)

            # Create a single compressed chunk
            compressed_chunk = Chunk(
                document_id="compressed",
                content=summary,
                token_count=compressed_tokens,
            )
            compressed_doc = RetrievalResult(
                chunk=compressed_chunk,
                score=max(d.score for d in documents) if documents else 0.0,
                retriever_name="llm_compressed",
            )

            return CompressedContext(
                documents=[compressed_doc],
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
                compression_method=self.compressor_name,
            )

        except Exception as e:
            logger.error("llm_compression_failed", error=str(e))
            return CompressedContext(
                documents=documents,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                compression_ratio=1.0,
                compression_method="passthrough",
            )
