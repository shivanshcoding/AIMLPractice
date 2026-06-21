"""
Reranker Manager.

Creates rerankers from config and manages their lifecycle.
"""

from __future__ import annotations

from config.settings import ModelConfig, get_settings
from src.core.interfaces import BaseReranker
from src.core.exceptions import ConfigurationError
from src.reranking.bge_reranker import BGEReranker
from src.reranking.cross_encoder import CrossEncoderReranker


def create_reranker(config: ModelConfig) -> BaseReranker:
    """Create a reranker from config."""
    model_lower = config.model.lower()

    if "bge-reranker" in model_lower or "bge_reranker" in model_lower:
        return BGEReranker(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            normalize=config.normalize,
        )
    else:
        return CrossEncoderReranker(
            model=config.model,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            normalize=config.normalize,
        )


def get_primary_reranker() -> BaseReranker:
    """Get the primary reranker from settings."""
    settings = get_settings()
    if settings.models is None:
        raise ConfigurationError("Models config not loaded.")
    return create_reranker(settings.models.reranker_model)


def get_all_rerankers() -> dict[str, BaseReranker]:
    """Get all configured rerankers for benchmarking."""
    settings = get_settings()
    if settings.models is None:
        raise ConfigurationError("Models config not loaded.")

    rerankers = {"primary": create_reranker(settings.models.reranker_model)}
    for name, config in settings.models.alternative_rerankers.items():
        rerankers[name] = create_reranker(config)

    return rerankers
