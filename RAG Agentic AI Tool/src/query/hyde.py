"""
HyDE — Hypothetical Document Embeddings.

Generates a hypothetical answer to the query, then embeds that answer
for retrieval. Particularly effective when query language differs
significantly from document language.
"""

from __future__ import annotations

import structlog

from src.core.interfaces import BaseLLM

logger = structlog.get_logger(__name__)

HYDE_PROMPT = """Please write a short passage that would answer the following question.
The passage should be informative and factual, written as if it were extracted from a
relevant document. Keep it to 2-3 paragraphs.

Question: {query}

Passage:"""


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE).

    Generates a hypothetical answer document, which is then embedded
    and used as the retrieval query instead of the original question.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query.

        Args:
            query: The original query.

        Returns:
            A hypothetical passage answering the query.
        """
        try:
            prompt = HYDE_PROMPT.format(query=query)
            response = self._llm.generate(prompt, temperature=0.7, max_tokens=512)
            hypothetical = response.strip()

            logger.debug(
                "hyde_generated",
                query=query[:50],
                hypothetical_length=len(hypothetical),
            )
            return hypothetical

        except Exception as e:
            logger.warning("hyde_generation_failed", error=str(e))
            return query  # Fall back to original query
