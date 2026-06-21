"""
LangSmith Integration.

Configures LangSmith tracing for all LangGraph nodes and LLM calls.
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing from settings.

    Returns True if tracing was enabled.
    """
    settings = get_settings()

    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        logger.info("langsmith_tracing_disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info(
        "langsmith_tracing_enabled",
        project=settings.langsmith_project,
    )
    return True


def get_tracing_config() -> dict[str, Any]:
    """Get LangSmith run config for manual trace attachment."""
    settings = get_settings()
    if settings.langsmith_tracing:
        return {
            "callbacks": [],
            "tags": ["rag-engine"],
            "metadata": {"project": settings.langsmith_project},
        }
    return {}
