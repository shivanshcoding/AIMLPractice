"""
Pure-Python Information Retrieval Metrics.

MRR, NDCG, Recall@K, Precision@K — no LLM judge required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant documents found in top-K."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for doc_id in top_k if doc_id in relevant) / k


def mrr(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant document."""
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of relevant document IDs.
        k: Cutoff.
        relevance_scores: Optional per-document relevance scores (default: binary).
    """
    relevant = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i in range(min(k, len(retrieved_ids))):
        doc_id = retrieved_ids[i]
        if relevance_scores:
            rel = relevance_scores.get(doc_id, 0.0)
        else:
            rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    if relevance_scores:
        ideal_rels = sorted(
            [relevance_scores.get(rid, 1.0) for rid in relevant_ids],
            reverse=True,
        )
    else:
        ideal_rels = [1.0] * len(relevant_ids)

    idcg = 0.0
    for i in range(min(k, len(ideal_rels))):
        idcg += ideal_rels[i] / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_all_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compute all IR metrics at once.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of relevant (ground-truth) document IDs.
        k_values: List of K values for @K metrics.

    Returns:
        Dict with all computed metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: dict[str, Any] = {
        "mrr": mrr(retrieved_ids, relevant_ids),
    }

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    return results
