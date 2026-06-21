"""
LangGraph Conditional Edge Logic.

Determines routing between nodes based on state.
"""

from __future__ import annotations

from typing import Literal

import structlog

from src.agent.state import RetrievalState

logger = structlog.get_logger(__name__)


def should_retry(state: RetrievalState) -> Literal["self_correction", "compression"]:
    """
    Conditional edge after confidence evaluation.

    Routes to self-correction if confidence is below threshold
    and max iterations haven't been reached.
    """
    confidence = state.get("confidence_score", 0.0)
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    # Default threshold
    threshold = 0.7

    if confidence < threshold and iteration < max_iterations:
        logger.info(
            "routing_to_self_correction",
            confidence=round(confidence, 3),
            iteration=iteration,
            max_iterations=max_iterations,
        )
        return "self_correction"

    if confidence < threshold:
        logger.warning(
            "low_confidence_max_iterations_reached",
            confidence=round(confidence, 3),
            iteration=iteration,
        )

    return "compression"
