"""
Query Classifier.

Classifies queries into types (factual, analytical, comparative, procedural)
to determine the optimal retrieval strategy.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.core.interfaces import BaseLLM
from src.core.models import QueryType

logger = structlog.get_logger(__name__)

CLASSIFICATION_PROMPT = """Classify the following query into exactly one category.

Categories:
- factual: Direct fact-seeking questions (who, what, when, where, how many)
- analytical: Questions requiring analysis, reasoning, or explanation (why, how does, explain)
- comparative: Questions comparing two or more things (vs, compare, difference, better)
- procedural: Step-by-step instructions or processes (how to, steps, guide, tutorial)

Query: {query}

Respond with ONLY the category name (factual, analytical, comparative, or procedural).
Category:"""


class QueryClassifier:
    """
    LLM-based query type classifier.

    Falls back to rule-based classification if LLM is unavailable.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm

    def classify(self, query: str) -> tuple[QueryType, float]:
        """
        Classify a query.

        Returns:
            Tuple of (QueryType, confidence_score).
        """
        # Try LLM classification first
        if self._llm:
            try:
                return self._classify_with_llm(query)
            except Exception as e:
                logger.warning("llm_classification_failed", error=str(e))

        # Fall back to rule-based
        return self._classify_with_rules(query)

    def _classify_with_llm(self, query: str) -> tuple[QueryType, float]:
        """Classify using the LLM."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        response = self._llm.generate(prompt, temperature=0.0, max_tokens=20)
        response = response.strip().lower()

        # Parse response
        for qt in QueryType:
            if qt.value in response:
                logger.debug("llm_classified", query=query[:50], type=qt.value)
                return qt, 0.9

        logger.warning("llm_classification_unparseable", response=response)
        return QueryType.FACTUAL, 0.5

    def _classify_with_rules(self, query: str) -> tuple[QueryType, float]:
        """Rule-based classification fallback."""
        query_lower = query.lower().strip()

        # Comparative patterns
        comparative_patterns = [
            r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b', r'\bcomparison\b',
            r'\bdifference\b', r'\bdiffer\b', r'\bbetter\b', r'\bworse\b',
            r'\badvantage\b', r'\bdisadvantage\b', r'\bpros\b', r'\bcons\b',
        ]
        for pattern in comparative_patterns:
            if re.search(pattern, query_lower):
                return QueryType.COMPARATIVE, 0.8

        # Procedural patterns
        procedural_patterns = [
            r'^how (to|do|can|should)\b', r'\bsteps?\b', r'\bguide\b',
            r'\btutorial\b', r'\bprocess\b', r'\bprocedure\b',
            r'\bimplement\b', r'\bsetup\b', r'\binstall\b', r'\bconfigure\b',
        ]
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                return QueryType.PROCEDURAL, 0.8

        # Analytical patterns
        analytical_patterns = [
            r'^why\b', r'^how does\b', r'\bexplain\b', r'\banalyz[es]\b',
            r'\breason\b', r'\bcause\b', r'\bimpact\b', r'\beffect\b',
            r'\bimplication\b', r'\bsignificance\b',
        ]
        for pattern in analytical_patterns:
            if re.search(pattern, query_lower):
                return QueryType.ANALYTICAL, 0.7

        # Default to factual
        return QueryType.FACTUAL, 0.6
