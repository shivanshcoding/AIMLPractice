"""
LangGraph Retrieval Workflow Builder.

Assembles the full retrieval StateGraph with all nodes and conditional edges.
Compiles into an executable graph with LangSmith tracing.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from src.agent.edges import should_retry
from src.agent.nodes import RetrievalNodes
from src.agent.state import RetrievalState
from src.core.interfaces import BaseCompressor, BaseEmbedder, BaseLLM, BaseReranker
from src.core.models import RetrievalContext
from src.indexing.bm25_store import BM25Store
from src.indexing.hybrid import HybridFusion
from src.indexing.qdrant_store import QdrantStore
from src.query.transformer import QueryTransformer

logger = structlog.get_logger(__name__)


class RetrievalGraph:
    """
    Builds and executes the LangGraph retrieval workflow.

    Graph topology:
        QueryAnalysis → RetrieverExecution → Reranking → ConfidenceCheck
            ↓ (low confidence)                              ↓ (sufficient)
        SelfCorrection → RetrieverExecution             Compression → END
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        qdrant_store: QdrantStore,
        bm25_store: BM25Store,
        reranker: BaseReranker | None = None,
        compressor: BaseCompressor | None = None,
        llm: BaseLLM | None = None,
        query_transformer: QueryTransformer | None = None,
        fusion: HybridFusion | None = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        confidence_threshold: float = 0.7,
        max_iterations: int = 3,
    ) -> None:
        self._nodes = RetrievalNodes(
            embedder=embedder,
            qdrant_store=qdrant_store,
            bm25_store=bm25_store,
            reranker=reranker,
            compressor=compressor,
            llm=llm,
            query_transformer=query_transformer,
            fusion=fusion,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
        )

        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph."""
        builder = StateGraph(RetrievalState)

        # Add nodes
        builder.add_node("query_analysis", self._nodes.query_analysis_node)
        builder.add_node("retriever_execution", self._nodes.retriever_execution_node)
        builder.add_node("reranking", self._nodes.reranking_node)
        builder.add_node("confidence_check", self._nodes.confidence_evaluation_node)
        builder.add_node("compression", self._nodes.compression_node)
        builder.add_node("self_correction", self._nodes.self_correction_node)

        # Set entry point
        builder.set_entry_point("query_analysis")

        # Add edges (linear flow)
        builder.add_edge("query_analysis", "retriever_execution")
        builder.add_edge("retriever_execution", "reranking")
        builder.add_edge("reranking", "confidence_check")

        # Conditional edge: confidence check → self_correction OR compression
        builder.add_conditional_edges(
            "confidence_check",
            should_retry,
            {
                "self_correction": "self_correction",
                "compression": "compression",
            },
        )

        # Self-correction loops back to retrieval
        builder.add_edge("self_correction", "retriever_execution")

        # Compression → END
        builder.add_edge("compression", END)

        # Compile
        compiled = builder.compile()
        logger.info("retrieval_graph_compiled")
        return compiled

    def invoke(
        self,
        query: str,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RetrievalContext:
        """
        Execute the retrieval workflow.

        Args:
            query: The user query.
            metadata_filters: Optional metadata filters for retrieval.

        Returns:
            RetrievalContext with all results and metrics.
        """
        start = time.perf_counter()

        # Initialize state
        initial_state: RetrievalState = {
            "query": query,
            "metadata_filters": metadata_filters or {},
            "query_analysis": None,
            "retrieval_strategy": "hybrid",
            "retrieved_documents": [],
            "reranked_documents": [],
            "compressed_context": None,
            "confidence_score": 0.0,
            "iteration_count": 0,
            "max_iterations": 3,
            "retrieval_latency_ms": 0.0,
            "reranking_latency_ms": 0.0,
            "compression_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "total_token_count": 0,
            "errors": [],
        }

        # Execute graph
        final_state = self._graph.invoke(initial_state)
        total_latency = (time.perf_counter() - start) * 1000

        # Build result
        documents = final_state.get("reranked_documents") or final_state.get(
            "retrieved_documents", []
        )

        context = RetrievalContext(
            query=query,
            query_analysis=final_state.get("query_analysis") or _default_analysis(query),
            documents=documents,
            compressed_context=final_state.get("compressed_context"),
            confidence_score=final_state.get("confidence_score", 0.0),
            retrieval_strategy=final_state.get("retrieval_strategy", "hybrid"),
            retrieval_iterations=final_state.get("iteration_count", 1),
            total_latency_ms=total_latency,
            retrieval_latency_ms=final_state.get("retrieval_latency_ms", 0.0),
            reranking_latency_ms=final_state.get("reranking_latency_ms", 0.0),
            compression_latency_ms=final_state.get("compression_latency_ms", 0.0),
            total_token_count=final_state.get("total_token_count", 0),
        )

        logger.info(
            "retrieval_complete",
            query=query[:80],
            documents=len(documents),
            confidence=round(context.confidence_score, 3),
            iterations=context.retrieval_iterations,
            latency_ms=round(total_latency, 1),
        )

        return context

    @property
    def graph(self) -> Any:
        """Access the compiled LangGraph for visualization."""
        return self._graph


def _default_analysis(query: str) -> Any:
    """Create a default QueryAnalysis when none is available."""
    from src.core.models import QueryAnalysis, QueryType, RetrieverType

    return QueryAnalysis(
        original_query=query,
        query_type=QueryType.UNKNOWN,
        transformed_queries=[query],
        retriever_type=RetrieverType.HYBRID,
    )
