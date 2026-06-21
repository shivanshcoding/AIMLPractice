"""
Configuration Settings Loader.

Loads all YAML config files and environment variables into typed Pydantic models.
Provides a single `get_settings()` accessor for the entire application.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# =============================================================================
# Resolve config directory relative to project root
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the config directory."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Model Configuration Schemas
# =============================================================================
class ModelConfig(BaseModel):
    """Configuration for a single model."""

    provider: str = "huggingface"
    model: str
    device: str = "cuda"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    dense_dim: int = 1024
    use_sparse: bool = False
    use_colbert: bool = False
    normalize: bool = True
    # For remote API providers (vLLM, OpenAI, etc.)
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class ModelsConfig(BaseModel):
    """All model configurations."""

    embedding_model: ModelConfig
    reranker_model: ModelConfig
    generation_model: ModelConfig
    query_expansion_model: ModelConfig
    evaluation_model: ModelConfig
    alternative_embeddings: dict[str, ModelConfig] = Field(default_factory=dict)
    alternative_rerankers: dict[str, ModelConfig] = Field(default_factory=dict)


# =============================================================================
# Retrieval Configuration Schemas
# =============================================================================
class QdrantConfig(BaseModel):
    """Qdrant vector store configuration."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    mode: str = "memory"  # memory | disk | server
    collection_name: str = "rag_documents"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    hnsw_config: dict[str, Any] = Field(
        default_factory=lambda: {"m": 16, "ef_construct": 100, "on_disk": False}
    )


class BM25Config(BaseModel):
    """BM25 sparse index configuration."""

    persist_path: str = "data/bm25_index.pkl"
    tokenizer: str = "word"
    lowercase: bool = True
    remove_stopwords: bool = True


class StrategyConfig(BaseModel):
    """Per-query-type retrieval strategy."""

    retriever: str = "hybrid"
    query_transform: str = "none"
    rerank: bool = True


class FusionConfig(BaseModel):
    """Fusion method configuration."""

    method: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class MetadataConfig(BaseModel):
    """Metadata filtering configuration."""

    filterable_fields: list[str] = Field(
        default_factory=lambda: [
            "source", "department", "date", "version", "tags", "access_level"
        ]
    )
    metadata_boost_enabled: bool = True
    boost_weights: dict[str, float] = Field(
        default_factory=lambda: {"department": 0.1, "recency": 0.05}
    )


class ContextOptimizationConfig(BaseModel):
    """Long-context optimization configuration."""

    lost_in_middle_mitigation: bool = True
    context_packing: bool = True
    max_context_tokens: int = 4096


class RetrievalDefaults(BaseModel):
    """Default retrieval parameters."""

    default_top_k: int = 10
    rerank_top_k: int = 5
    max_retrieval_iterations: int = 3
    confidence_threshold: float = 0.7
    strategy_map: dict[str, StrategyConfig] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    """Complete retrieval configuration."""

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    retrieval: RetrievalDefaults = Field(default_factory=RetrievalDefaults)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    context_optimization: ContextOptimizationConfig = Field(
        default_factory=ContextOptimizationConfig
    )


# =============================================================================
# Chunking Configuration Schemas
# =============================================================================
class FixedChunkConfig(BaseModel):
    """Fixed chunking parameters."""

    chunk_size: int = 512
    chunk_overlap: int = 64


class RecursiveChunkConfig(BaseModel):
    """Recursive character chunking parameters."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


class SemanticChunkConfig(BaseModel):
    """Semantic chunking parameters."""

    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold: int = 95
    min_chunk_size: int = 100
    max_chunk_size: int = 2000


class ParentChildChunkConfig(BaseModel):
    """Parent-child hierarchical chunking parameters."""

    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 500
    child_chunk_overlap: int = 50


class DocumentAwareChunkConfig(BaseModel):
    """Document-aware structural chunking parameters."""

    respect_headings: bool = True
    respect_paragraphs: bool = True
    respect_lists: bool = True
    respect_tables: bool = True
    fallback_strategy: str = "recursive"
    max_chunk_size: int = 1500
    min_chunk_size: int = 100


class AutoSelectionRule(BaseModel):
    """Auto-selection heuristic rule."""

    condition: str
    strategy: str


class AutoSelectionConfig(BaseModel):
    """Auto-selection configuration."""

    rules: list[AutoSelectionRule] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    """Complete chunking configuration."""

    default_strategy: str = "auto"
    fixed: FixedChunkConfig = Field(default_factory=FixedChunkConfig)
    recursive: RecursiveChunkConfig = Field(default_factory=RecursiveChunkConfig)
    semantic: SemanticChunkConfig = Field(default_factory=SemanticChunkConfig)
    parent_child: ParentChildChunkConfig = Field(default_factory=ParentChildChunkConfig)
    document_aware: DocumentAwareChunkConfig = Field(default_factory=DocumentAwareChunkConfig)
    auto_selection: AutoSelectionConfig = Field(default_factory=AutoSelectionConfig)


# =============================================================================
# Top-Level Application Settings
# =============================================================================
class Settings(BaseSettings):
    """
    Application settings aggregating all configs.

    Priority: env vars > .env file > YAML defaults
    """

    # --- Environment-level settings ---
    app_name: str = "RAG Retrieval Engine"
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"

    # --- LangSmith Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag-retrieval-engine"
    langsmith_tracing: bool = False

    # --- OpenTelemetry ---
    otel_endpoint: str | None = None
    otel_service_name: str = "rag-retrieval-engine"

    # --- Loaded configs (populated in __init__) ---
    models: ModelsConfig | None = None
    retrieval_config: RetrievalConfig | None = None
    chunking_config: ChunkingConfig | None = None

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Load YAML configs
        try:
            models_raw = _load_yaml("models.yaml")
            self.models = ModelsConfig(**models_raw)
        except FileNotFoundError:
            pass

        try:
            retrieval_raw = _load_yaml("retrieval.yaml")
            self.retrieval_config = RetrievalConfig(**retrieval_raw)
        except FileNotFoundError:
            pass

        try:
            chunking_raw = _load_yaml("chunking.yaml")
            self.chunking_config = ChunkingConfig(**chunking_raw)
        except FileNotFoundError:
            pass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
