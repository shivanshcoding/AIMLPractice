"""
Component Registry & Factory.

Central registry that maps string names to component implementations.
Enables dependency injection and runtime component swapping via config.
"""

from __future__ import annotations

import structlog
from typing import Any, TypeVar, Type

from src.core.interfaces import (
    BaseChunker,
    BaseCompressor,
    BaseEmbedder,
    BaseLLM,
    BaseReranker,
    BaseRetriever,
)
from src.core.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ComponentRegistry:
    """
    Registry for all pluggable RAG components.

    Usage:
        registry = ComponentRegistry()
        registry.register_chunker("recursive", RecursiveChunker)
        chunker = registry.get_chunker("recursive", **config)
    """

    def __init__(self) -> None:
        self._chunkers: dict[str, type[BaseChunker]] = {}
        self._embedders: dict[str, type[BaseEmbedder]] = {}
        self._retrievers: dict[str, type[BaseRetriever]] = {}
        self._rerankers: dict[str, type[BaseReranker]] = {}
        self._compressors: dict[str, type[BaseCompressor]] = {}
        self._llms: dict[str, type[BaseLLM]] = {}
        # Instance cache for singletons (e.g., loaded models)
        self._instances: dict[str, Any] = {}

    # --- Registration Methods ---
    def register_chunker(self, name: str, cls: type[BaseChunker]) -> None:
        """Register a chunker implementation."""
        self._chunkers[name] = cls
        logger.debug("registered_chunker", name=name, cls=cls.__name__)

    def register_embedder(self, name: str, cls: type[BaseEmbedder]) -> None:
        """Register an embedder implementation."""
        self._embedders[name] = cls
        logger.debug("registered_embedder", name=name, cls=cls.__name__)

    def register_retriever(self, name: str, cls: type[BaseRetriever]) -> None:
        """Register a retriever implementation."""
        self._retrievers[name] = cls
        logger.debug("registered_retriever", name=name, cls=cls.__name__)

    def register_reranker(self, name: str, cls: type[BaseReranker]) -> None:
        """Register a reranker implementation."""
        self._rerankers[name] = cls
        logger.debug("registered_reranker", name=name, cls=cls.__name__)

    def register_compressor(self, name: str, cls: type[BaseCompressor]) -> None:
        """Register a compressor implementation."""
        self._compressors[name] = cls
        logger.debug("registered_compressor", name=name, cls=cls.__name__)

    def register_llm(self, name: str, cls: type[BaseLLM]) -> None:
        """Register an LLM implementation."""
        self._llms[name] = cls
        logger.debug("registered_llm", name=name, cls=cls.__name__)

    # --- Factory Methods ---
    def _get_or_create(
        self,
        registry: dict[str, type[T]],
        name: str,
        component_type: str,
        singleton: bool = False,
        **kwargs: Any,
    ) -> T:
        """Get or create a component instance."""
        if name not in registry:
            available = list(registry.keys())
            raise ConfigurationError(
                f"Unknown {component_type}: '{name}'. Available: {available}"
            )

        cache_key = f"{component_type}:{name}"
        if singleton and cache_key in self._instances:
            return self._instances[cache_key]

        cls = registry[name]
        instance = cls(**kwargs)

        if singleton:
            self._instances[cache_key] = instance

        logger.info(
            "created_component",
            component_type=component_type,
            name=name,
            singleton=singleton,
        )
        return instance

    def get_chunker(self, name: str, **kwargs: Any) -> BaseChunker:
        """Get a chunker instance by name."""
        return self._get_or_create(self._chunkers, name, "chunker", **kwargs)

    def get_embedder(self, name: str, singleton: bool = True, **kwargs: Any) -> BaseEmbedder:
        """Get an embedder instance by name (singleton by default to reuse GPU memory)."""
        return self._get_or_create(
            self._embedders, name, "embedder", singleton=singleton, **kwargs
        )

    def get_retriever(self, name: str, **kwargs: Any) -> BaseRetriever:
        """Get a retriever instance by name."""
        return self._get_or_create(self._retrievers, name, "retriever", **kwargs)

    def get_reranker(self, name: str, singleton: bool = True, **kwargs: Any) -> BaseReranker:
        """Get a reranker instance by name (singleton by default)."""
        return self._get_or_create(
            self._rerankers, name, "reranker", singleton=singleton, **kwargs
        )

    def get_compressor(self, name: str, **kwargs: Any) -> BaseCompressor:
        """Get a compressor instance by name."""
        return self._get_or_create(self._compressors, name, "compressor", **kwargs)

    def get_llm(self, name: str, singleton: bool = True, **kwargs: Any) -> BaseLLM:
        """Get an LLM instance by name (singleton by default)."""
        return self._get_or_create(
            self._llms, name, "llm", singleton=singleton, **kwargs
        )

    # --- Introspection ---
    def list_chunkers(self) -> list[str]:
        """List registered chunker names."""
        return list(self._chunkers.keys())

    def list_embedders(self) -> list[str]:
        """List registered embedder names."""
        return list(self._embedders.keys())

    def list_retrievers(self) -> list[str]:
        """List registered retriever names."""
        return list(self._retrievers.keys())

    def list_rerankers(self) -> list[str]:
        """List registered reranker names."""
        return list(self._rerankers.keys())

    def list_compressors(self) -> list[str]:
        """List registered compressor names."""
        return list(self._compressors.keys())

    def list_llms(self) -> list[str]:
        """List registered LLM names."""
        return list(self._llms.keys())

    def clear_cache(self) -> None:
        """Clear all cached singleton instances."""
        self._instances.clear()
        logger.info("cleared_component_cache")


# Global registry singleton
_global_registry: ComponentRegistry | None = None


def get_registry() -> ComponentRegistry:
    """Get or create the global component registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry
