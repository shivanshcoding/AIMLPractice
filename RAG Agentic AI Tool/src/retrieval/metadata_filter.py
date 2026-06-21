"""
Metadata-Aware Filtering.

Constructs metadata filters from query analysis and applies
metadata-based score boosting.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


def extract_metadata_filters(query: str) -> dict[str, Any]:
    """
    Extract metadata filters from query text.

    Detects patterns like:
      - "from department X" → {"department": "X"}
      - "tagged with X" → {"tags": ["X"]}
      - "version X" → {"version": "X"}
      - "source: X" → {"source": "X"}
    """
    filters: dict[str, Any] = {}

    # Department extraction
    dept_match = re.search(
        r'(?:from|department|dept)[:\s]+(["\']?)(\w+)\1',
        query,
        re.IGNORECASE,
    )
    if dept_match:
        filters["department"] = dept_match.group(2)

    # Source extraction
    source_match = re.search(
        r'(?:source|from file)[:\s]+(["\']?)([^\s,]+)\1',
        query,
        re.IGNORECASE,
    )
    if source_match:
        filters["source"] = source_match.group(2)

    # Version extraction
    version_match = re.search(
        r'(?:version|v)[:\s]*(["\']?)(\d+[\.\d]*)\1',
        query,
        re.IGNORECASE,
    )
    if version_match:
        filters["version"] = version_match.group(2)

    # Access level extraction
    access_match = re.search(
        r'(?:access|level)[:\s]+(["\']?)(\w+)\1',
        query,
        re.IGNORECASE,
    )
    if access_match:
        filters["access_level"] = access_match.group(2)

    if filters:
        logger.info("metadata_filters_extracted", filters=filters)

    return filters


def apply_metadata_boost(
    scores: list[float],
    metadata_list: list[dict[str, Any]],
    boost_weights: dict[str, float] | None = None,
) -> list[float]:
    """
    Apply metadata-based score boosting.

    Boosts scores based on metadata matches (e.g., recency, department match).
    """
    settings = get_settings()
    if boost_weights is None and settings.retrieval_config:
        boost_weights = settings.retrieval_config.metadata.boost_weights

    if not boost_weights:
        return scores

    boosted = list(scores)
    for i, meta in enumerate(metadata_list):
        boost = 0.0
        for field, weight in boost_weights.items():
            if field == "recency" and "date" in meta and meta["date"]:
                # More recent = higher boost (simplified)
                boost += weight
            elif field in meta and meta[field]:
                boost += weight
        boosted[i] = scores[i] * (1.0 + boost)

    return boosted
