"""
Query Expansion & Multi-Query Generation.

Generates diverse reformulations and expansions of the original query
to improve retrieval recall.
"""

from __future__ import annotations

import structlog

from src.core.interfaces import BaseLLM

logger = structlog.get_logger(__name__)

MULTI_QUERY_PROMPT = """You are an AI assistant helping to improve information retrieval.
Given the following query, generate {n} different reformulations that capture the same intent
but use different wording, perspectives, or levels of specificity.

Original query: {query}

Return ONLY the reformulated queries, one per line, numbered 1-{n}.
Do not include any other text."""

QUERY_EXPANSION_PROMPT = """Given the following search query, expand it by adding relevant
synonyms, related terms, and context that would help find relevant documents.

Original query: {query}

Return a single expanded query that incorporates the original query with additional
relevant terms. Do not change the meaning, only broaden the recall.

Expanded query:"""


class QueryExpander:
    """
    Generates multi-query reformulations and expanded queries.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """
        Generate N diverse reformulations of the query.

        Args:
            query: Original query.
            n: Number of reformulations.

        Returns:
            List of reformulated queries (includes original).
        """
        try:
            prompt = MULTI_QUERY_PROMPT.format(query=query, n=n)
            response = self._llm.generate(prompt, temperature=0.7, max_tokens=512)

            queries = [query]  # Always include original
            for line in response.strip().split("\n"):
                line = line.strip()
                # Remove numbering
                if line and line[0].isdigit():
                    line = line.lstrip("0123456789.)")
                    line = line.strip()
                if line and line != query:
                    queries.append(line)

            logger.debug("multi_query_generated", original=query[:50], count=len(queries))
            return queries[:n + 1]  # Original + N reformulations

        except Exception as e:
            logger.warning("multi_query_failed", error=str(e))
            return [query]

    def expand_query(self, query: str) -> str:
        """
        Expand a query with related terms and context.

        Args:
            query: Original query.

        Returns:
            Expanded query string.
        """
        try:
            prompt = QUERY_EXPANSION_PROMPT.format(query=query)
            response = self._llm.generate(prompt, temperature=0.3, max_tokens=256)
            expanded = response.strip()

            if expanded:
                logger.debug(
                    "query_expanded",
                    original_len=len(query),
                    expanded_len=len(expanded),
                )
                return expanded

            return query

        except Exception as e:
            logger.warning("query_expansion_failed", error=str(e))
            return query
