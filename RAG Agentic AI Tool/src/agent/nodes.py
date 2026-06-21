"""
LangGraph Node Implementations.

Each node is a pure function: (state) → state_update.
Nodes have a single responsibility and are independently testable.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.agent.state import RetrievalState
from src.core.interfaces import BaseCompressor, BaseEmbedder, BaseLLM, BaseReranker
from src.core.models import (
    CompressedContext,
    QueryAnalysis,
    QueryType,
    RetrieverType,
    RetrievalResult,
)
from src.indexing.bm25_store import BM25Store
from src.indexing.hybrid import HybridFusion
from src.indexing.qdrant_store import QdrantStore
from src.query.transformer import QueryTransformer

logger = structlog.get_logger(__name__)


class RetrievalNodes:
    """
    Container for all LangGraph node implementations.

    Each method is a graph node: accepts state, returns state update dict.
    Dependencies are injected via __init__.
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
        self._embedder = embedder
        self._qdrant = qdrant_store
        self._bm25 = bm25_store
        self._reranker = reranker
        self._compressor = compressor
        self._llm = llm
        self._query_transformer = query_transformer or QueryTransformer(llm=llm)
        self._fusion = fusion or HybridFusion.from_config()
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k
        self._confidence_threshold = confidence_threshold
        self._max_iterations = max_iterations

    def query_analysis_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 1: Analyze and classify the query.

        Determines query type and transformation strategy.
        """
        query = state["query"]
        logger.info("node_query_analysis", query=query[:80])

        try:
            analysis = self._query_transformer.analyze(query)
            return {
                "query_analysis": analysis,
                "retrieval_strategy": analysis.retriever_type.value,
                "iteration_count": state.get("iteration_count", 0),
                "max_iterations": self._max_iterations,
                "errors": state.get("errors", []),
            }
        except Exception as e:
            logger.error("query_analysis_failed", error=str(e))
            # Graceful fallback
            fallback = QueryAnalysis(
                original_query=query,
                query_type=QueryType.FACTUAL,
                classification_confidence=0.5,
                transformed_queries=[query],
                retriever_type=RetrieverType.HYBRID,
            )
            return {
                "query_analysis": fallback,
                "retrieval_strategy": "hybrid",
                "errors": state.get("errors", []) + [f"Analysis failed: {e}"],
            }

    def retriever_execution_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 2: Execute retrieval.

        Runs dense, sparse, or hybrid retrieval based on the analysis.
        Supports multi-query by running each transformed query and merging results.
        """
        start = time.perf_counter()
        analysis = state.get("query_analysis")
        filters = state.get("metadata_filters", {})

        if analysis is None:
            return {"retrieved_documents": [], "errors": ["No query analysis"]}

        all_results: list[RetrievalResult] = []
        queries = analysis.transformed_queries or [analysis.original_query]

        # If HyDE, use hypothetical document embedding for the first query
        primary_query = analysis.original_query
        if analysis.hypothetical_document:
            primary_query = analysis.hypothetical_document

        for i, query_text in enumerate(queries):
            # Use hypothetical doc for first query if HyDE
            embed_text = primary_query if i == 0 and analysis.hypothetical_document else query_text

            try:
                # Dense retrieval
                query_vector = self._embedder.embed_query(embed_text)
                dense_results = self._qdrant.search_dense(
                    query_vector=query_vector,
                    top_k=self._top_k,
                    filters=filters if filters else None,
                )

                # Sparse retrieval (BM25)
                sparse_results = self._bm25.search(
                    query=query_text,  # Always use original text for BM25
                    top_k=self._top_k,
                    filters=filters if filters else None,
                )

                # Hybrid fusion
                if analysis.retriever_type == RetrieverType.HYBRID:
                    fused = self._fusion.fuse(dense_results, sparse_results, self._top_k)
                    all_results.extend(fused)
                elif analysis.retriever_type == RetrieverType.DENSE:
                    all_results.extend(dense_results)
                else:
                    all_results.extend(sparse_results)

            except Exception as e:
                logger.error("retrieval_failed", query_index=i, error=str(e))

        # Deduplicate by chunk_id, keeping highest score
        seen: dict[str, RetrievalResult] = {}
        for result in all_results:
            cid = result.chunk.chunk_id
            if cid not in seen or result.score > seen[cid].score:
                seen[cid] = result

        # Sort by score descending
        unique_results = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        unique_results = unique_results[: self._top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "node_retrieval_complete",
            results_count=len(unique_results),
            queries_count=len(queries),
            latency_ms=round(elapsed_ms, 1),
        )

        return {
            "retrieved_documents": unique_results,
            "retrieval_latency_ms": elapsed_ms,
        }

    def reranking_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 3: Rerank retrieved documents.

        Uses the configured reranker (BGE Reranker v2 M3 by default).
        """
        start = time.perf_counter()
        documents = state.get("retrieved_documents", [])
        query = state.get("query", "")

        if not documents or self._reranker is None:
            return {
                "reranked_documents": documents,
                "reranking_latency_ms": 0.0,
            }

        try:
            reranked = self._reranker.rerank(
                query=query,
                documents=documents,
                top_k=self._rerank_top_k,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "node_reranking_complete",
                input_count=len(documents),
                output_count=len(reranked),
                latency_ms=round(elapsed_ms, 1),
            )

            return {
                "reranked_documents": reranked,
                "reranking_latency_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error("reranking_failed", error=str(e))
            return {
                "reranked_documents": documents[: self._rerank_top_k],
                "reranking_latency_ms": 0.0,
                "errors": state.get("errors", []) + [f"Reranking failed: {e}"],
            }

    def confidence_evaluation_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 4: Evaluate retrieval confidence.

        Scores confidence based on:
          - Top score magnitude
          - Score distribution (gap between top and bottom)
          - Number of results
        """
        documents = state.get("reranked_documents", state.get("retrieved_documents", []))

        if not documents:
            return {"confidence_score": 0.0}

        scores = [d.score for d in documents]

        # Confidence factors
        top_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0.0

        # Heuristic confidence calculation
        # High top score → high confidence
        score_confidence = min(top_score / 1.0, 1.0) if top_score > 0 else 0.0

        # Good separation between relevant and irrelevant → high confidence
        separation_confidence = min(score_range / 0.5, 1.0) if score_range > 0 else 0.3

        # Having enough results → higher confidence
        count_confidence = min(len(documents) / 3, 1.0)

        # Weighted combination
        confidence = (
            0.5 * score_confidence
            + 0.3 * separation_confidence
            + 0.2 * count_confidence
        )

        logger.info(
            "node_confidence_evaluated",
            confidence=round(confidence, 3),
            top_score=round(top_score, 3),
            avg_score=round(avg_score, 3),
            num_results=len(documents),
        )

        return {"confidence_score": confidence}

    def compression_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 5: Compress context.

        Reduces token count using the configured compressor.
        """
        start = time.perf_counter()
        documents = state.get("reranked_documents", state.get("retrieved_documents", []))
        query = state.get("query", "")

        if not documents:
            return {
                "compressed_context": None,
                "compression_latency_ms": 0.0,
            }

        if self._compressor is None:
            # No compression configured — pass through
            total_tokens = sum(d.chunk.token_count for d in documents)
            compressed = CompressedContext(
                documents=documents,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                compression_ratio=1.0,
                compression_method="none",
            )
            return {
                "compressed_context": compressed,
                "compression_latency_ms": 0.0,
                "total_token_count": total_tokens,
            }

        try:
            compressed = self._compressor.compress(query=query, documents=documents)
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "node_compression_complete",
                original_tokens=compressed.original_token_count,
                compressed_tokens=compressed.compressed_token_count,
                ratio=round(compressed.compression_ratio, 2),
                latency_ms=round(elapsed_ms, 1),
            )

            return {
                "compressed_context": compressed,
                "compression_latency_ms": elapsed_ms,
                "total_token_count": compressed.compressed_token_count,
            }

        except Exception as e:
            logger.error("compression_failed", error=str(e))
            total_tokens = sum(d.chunk.token_count for d in documents)
            return {
                "compressed_context": CompressedContext(
                    documents=documents,
                    original_token_count=total_tokens,
                    compressed_token_count=total_tokens,
                    compression_ratio=1.0,
                    compression_method="passthrough",
                ),
                "compression_latency_ms": 0.0,
            }

    def self_correction_node(self, state: RetrievalState) -> dict[str, Any]:
        """
        Node 6: Self-correction on low confidence.

        Reformulates the query, increases retrieval depth, and retries.
        """
        iteration = state.get("iteration_count", 0) + 1
        query = state.get("query", "")

        logger.info(
            "node_self_correction",
            iteration=iteration,
            original_query=query[:50],
        )

        # Reformulate query using LLM if available
        new_queries = [query]
        if self._llm:
            try:
                reformulation_prompt = (
                    f"The following search query did not return confident results. "
                    f"Please rephrase it to be more specific and use different keywords.\n\n"
                    f"Original query: {query}\n\n"
                    f"Rephrased query:"
                )
                rephrased = self._llm.generate(
                    reformulation_prompt, temperature=0.5, max_tokens=128
                )
                new_queries = [rephrased.strip(), query]
            except Exception as e:
                logger.warning("self_correction_reformulation_failed", error=str(e))

        # Update analysis with new queries
        analysis = state.get("query_analysis")
        if analysis:
            analysis.transformed_queries = new_queries

        return {
            "query_analysis": analysis,
            "iteration_count": iteration,
            "retrieved_documents": [],  # Clear for retry
            "reranked_documents": [],
        }
