"""
LLM Factory — Creates LLM instances from config.

Reads models.yaml and instantiates the correct LLM client.
No source code changes needed to swap providers.
"""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import ModelConfig, get_settings
from src.core.exceptions import ConfigurationError
from src.core.interfaces import BaseLLM
from src.llm.hf_client import HFClient

logger = structlog.get_logger(__name__)

# Provider → Client class mapping
_PROVIDER_MAP: dict[str, type[BaseLLM]] = {
    "huggingface": HFClient,
    "local": HFClient,
}


def create_llm(config: ModelConfig) -> BaseLLM:
    """
    Create an LLM instance from a ModelConfig.

    Args:
        config: Model configuration from models.yaml.

    Returns:
        An initialized BaseLLM implementation.

    Raises:
        ConfigurationError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider not in _PROVIDER_MAP:
        raise ConfigurationError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Available: {list(_PROVIDER_MAP.keys())}"
        )

    # Local provider
    logger.info(
        "using_local_provider",
        provider=provider,
        model=config.model,
    )

    client_cls = _PROVIDER_MAP[provider]
    return client_cls(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )


def get_llm(role: str = "generation") -> BaseLLM:
    """
    Get an LLM instance by role from the global settings.

    Args:
        role: One of "generation", "query_expansion", "evaluation".

    Returns:
        An initialized BaseLLM for the specified role.
    """
    settings = get_settings()

    if settings.models is None:
        raise ConfigurationError("Models config not loaded. Check config/models.yaml.")

    role_map = {
        "generation": settings.models.generation_model,
        "query_expansion": settings.models.query_expansion_model,
        "evaluation": settings.models.evaluation_model,
    }

    if role not in role_map:
        raise ConfigurationError(
            f"Unknown LLM role: '{role}'. Available: {list(role_map.keys())}"
        )

    config = role_map[role]
    logger.info("creating_llm", role=role, model=config.model, provider=config.provider)
    return create_llm(config)
