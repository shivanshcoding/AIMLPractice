"""
Query Transformation Orchestrator.

Routes queries through the appropriate transformation pipeline
based on query classification.

Mapping:
  Factual     → direct retrieval (no transformation)
  Analytical  → step-back + multi-query
  Comparative → multi-query
  Procedural  → HyDE + step-back
"""

from __future__ import annotations

import structlog

from src.core.interfaces import BaseLLM
from src.core.models import QueryAnalysis, QueryTransformType, QueryType, RetrieverType
from src.query.classifier import QueryClassifier
from src.query.expander import QueryExpander
from src.query.hyde import HyDEGenerator
from src.query.step_back import StepBackGenerator

logger = structlog.get_logger(__name__)


class QueryTransformer:
    """
    Orchestrates query understanding and transformation.

    Classifies the query, applies the appropriate transformation,
    and returns a QueryAnalysis with transformed queries.
    """

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._classifier = QueryClassifier(llm=llm)
        self._expander = QueryExpander(llm) if llm else None
        self._hyde = HyDEGenerator(llm) if llm else None
        self._step_back = StepBackGenerator(llm) if llm else None

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Full query analysis: classify → transform → return QueryAnalysis.

        Args:
            query: The original user query.

        Returns:
            QueryAnalysis with classification and transformed queries.
        """
        # Step 1: Classify
        query_type, confidence = self._classifier.classify(query)

        # Step 2: Transform based on type
        transformed_queries = [query]
        transform_type = QueryTransformType.NONE
        hypothetical_doc = None

        if query_type == QueryType.FACTUAL:
            # No transformation needed for factual queries
            transform_type = QueryTransformType.NONE

        elif query_type == QueryType.ANALYTICAL:
            # Step-back + multi-query for analytical
            if self._step_back:
                step_back_q = self._step_back.generate_step_back_query(query)
                transformed_queries.append(step_back_q)
            if self._expander:
                multi = self._expander.multi_query(query, n=2)
                transformed_queries.extend(multi[1:])  # Skip original (already in list)
            transform_type = QueryTransformType.MULTI_QUERY

        elif query_type == QueryType.COMPARATIVE:
            # Multi-query for comparative (generate diverse perspectives)
            if self._expander:
                multi = self._expander.multi_query(query, n=3)
                transformed_queries = multi
            transform_type = QueryTransformType.MULTI_QUERY

        elif query_type == QueryType.PROCEDURAL:
            # HyDE + step-back for procedural
            if self._hyde:
                hypothetical_doc = self._hyde.generate_hypothetical_document(query)
            if self._step_back:
                step_back_q = self._step_back.generate_step_back_query(query)
                transformed_queries.append(step_back_q)
            transform_type = QueryTransformType.HYDE

        # Determine retriever type
        retriever_type = RetrieverType.HYBRID  # Default to hybrid for all types

        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            classification_confidence=confidence,
            transformed_queries=transformed_queries,
            query_transform_type=transform_type,
            retriever_type=retriever_type,
            hypothetical_document=hypothetical_doc,
        )

        logger.info(
            "query_analyzed",
            query=query[:80],
            type=query_type.value,
            confidence=confidence,
            num_transformed=len(transformed_queries),
            transform_type=transform_type.value,
        )

        return analysis
