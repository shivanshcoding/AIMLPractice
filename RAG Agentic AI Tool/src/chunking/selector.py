"""
Auto Chunking Strategy Selector.

Analyzes document structure features and selects the optimal
chunking strategy based on configurable heuristics.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from config.settings import get_settings
from src.core.interfaces import BaseChunker, BaseDocumentAnalyzer, BaseEmbedder
from src.core.models import (
    Chunk,
    ChunkingStrategy,
    Document,
    DocumentMetadata,
    DocumentStructure,
)
from src.chunking.base import BaseChunkerMixin
from src.chunking.fixed import FixedChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.semantic import SemanticChunker
from src.chunking.parent_child import ParentChildChunker
from src.chunking.document_aware import DocumentAwareChunker

logger = structlog.get_logger(__name__)


class DocumentAnalyzer(BaseDocumentAnalyzer):
    """Analyzes document structure to inform chunking strategy selection."""

    def analyze(self, document: Document) -> DocumentStructure:
        text = document.content
        lines = text.split("\n")

        # Count headings (markdown-style)
        heading_count = sum(1 for line in lines if re.match(r'^#{1,6}\s', line.strip()))

        # Count tables
        table_count = sum(1 for line in lines if line.strip().startswith("|"))
        # Rough table count (groups of consecutive | lines)
        table_count = max(1, table_count // 3) if table_count > 0 else 0

        # Count lists
        list_count = sum(
            1 for line in lines
            if re.match(r'^\s*[-*+•]\s|^\s*\d+\.\s', line.strip())
        )

        # Count code blocks
        code_block_count = text.count("```") // 2

        # Average paragraph length
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        avg_paragraph_length = (
            sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        )

        # Heading density (per 1000 chars)
        total_chars = len(text)
        heading_density = (heading_count / total_chars * 1000) if total_chars > 0 else 0

        # Check for hierarchical sections
        heading_levels = set()
        for line in lines:
            match = re.match(r'^(#{1,6})\s', line.strip())
            if match:
                heading_levels.add(len(match.group(1)))
        has_hierarchical = len(heading_levels) >= 2

        total_tokens = BaseChunkerMixin.count_tokens(text)

        structure = DocumentStructure(
            heading_count=heading_count,
            heading_density=heading_density,
            avg_paragraph_length=avg_paragraph_length,
            table_count=table_count,
            list_count=list_count,
            code_block_count=code_block_count,
            has_hierarchical_sections=has_hierarchical,
            total_characters=total_chars,
            total_tokens=total_tokens,
        )

        # Determine recommended strategy using heuristics
        structure.recommended_strategy = self._recommend_strategy(structure)
        return structure

    def _recommend_strategy(self, structure: DocumentStructure) -> ChunkingStrategy:
        """Apply heuristic rules to recommend a chunking strategy."""
        # Rule 1: Rich structure with tables → document-aware
        if structure.heading_density > 0.05 and structure.table_count > 0:
            return ChunkingStrategy.DOCUMENT_AWARE

        # Rule 2: Long paragraphs → semantic (need meaning-based splits)
        if structure.avg_paragraph_length > 500:
            return ChunkingStrategy.SEMANTIC

        # Rule 3: Hierarchical sections → parent-child
        if structure.has_hierarchical_sections and structure.heading_count >= 4:
            return ChunkingStrategy.PARENT_CHILD

        # Rule 4: Document-aware if lots of structural elements
        if structure.heading_count > 2 or structure.table_count > 0:
            return ChunkingStrategy.DOCUMENT_AWARE

        # Default: recursive (good general-purpose)
        return ChunkingStrategy.RECURSIVE


class ChunkingSelector:
    """
    Selects and executes the optimal chunking strategy for a document.

    Can either auto-select based on document analysis or use a specified strategy.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        **kwargs,
    ) -> None:
        self._analyzer = DocumentAnalyzer()
        self._embedder = embedder

        # Load config
        settings = get_settings()
        chunking_config = settings.chunking_config

        # Initialize all chunkers from config
        if chunking_config:
            self._chunkers: dict[ChunkingStrategy, BaseChunker] = {
                ChunkingStrategy.FIXED: FixedChunker(
                    chunk_size=chunking_config.fixed.chunk_size,
                    chunk_overlap=chunking_config.fixed.chunk_overlap,
                ),
                ChunkingStrategy.RECURSIVE: RecursiveChunker(
                    chunk_size=chunking_config.recursive.chunk_size,
                    chunk_overlap=chunking_config.recursive.chunk_overlap,
                    separators=chunking_config.recursive.separators,
                ),
                ChunkingStrategy.PARENT_CHILD: ParentChildChunker(
                    parent_chunk_size=chunking_config.parent_child.parent_chunk_size,
                    parent_chunk_overlap=chunking_config.parent_child.parent_chunk_overlap,
                    child_chunk_size=chunking_config.parent_child.child_chunk_size,
                    child_chunk_overlap=chunking_config.parent_child.child_chunk_overlap,
                ),
                ChunkingStrategy.DOCUMENT_AWARE: DocumentAwareChunker(
                    respect_headings=chunking_config.document_aware.respect_headings,
                    respect_paragraphs=chunking_config.document_aware.respect_paragraphs,
                    respect_lists=chunking_config.document_aware.respect_lists,
                    respect_tables=chunking_config.document_aware.respect_tables,
                    max_chunk_size=chunking_config.document_aware.max_chunk_size,
                    min_chunk_size=chunking_config.document_aware.min_chunk_size,
                ),
            }
            # Semantic chunker requires an embedder
            if embedder:
                self._chunkers[ChunkingStrategy.SEMANTIC] = SemanticChunker(
                    embedder=embedder,
                    breakpoint_threshold_type=chunking_config.semantic.breakpoint_threshold_type,
                    breakpoint_threshold=chunking_config.semantic.breakpoint_threshold,
                    min_chunk_size=chunking_config.semantic.min_chunk_size,
                    max_chunk_size=chunking_config.semantic.max_chunk_size,
                )
        else:
            # Fallback defaults
            self._chunkers = {
                ChunkingStrategy.FIXED: FixedChunker(),
                ChunkingStrategy.RECURSIVE: RecursiveChunker(),
                ChunkingStrategy.PARENT_CHILD: ParentChildChunker(),
                ChunkingStrategy.DOCUMENT_AWARE: DocumentAwareChunker(),
            }
            if embedder:
                self._chunkers[ChunkingStrategy.SEMANTIC] = SemanticChunker(embedder=embedder)

    def analyze(self, document: Document) -> DocumentStructure:
        """Analyze document structure."""
        return self._analyzer.analyze(document)

    def chunk(
        self,
        document: Document,
        strategy: ChunkingStrategy | str = ChunkingStrategy.AUTO,
    ) -> list[Chunk]:
        """
        Chunk a document using the specified or auto-selected strategy.

        Args:
            document: The document to chunk.
            strategy: Strategy to use, or "auto" for auto-selection.

        Returns:
            List of chunks.
        """
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)

        if strategy == ChunkingStrategy.AUTO:
            structure = self._analyzer.analyze(document)
            strategy = structure.recommended_strategy
            logger.info(
                "auto_selected_chunking_strategy",
                strategy=strategy.value,
                heading_count=structure.heading_count,
                avg_paragraph_length=structure.avg_paragraph_length,
                table_count=structure.table_count,
                has_hierarchical=structure.has_hierarchical_sections,
            )

        if strategy not in self._chunkers:
            logger.warning(
                "strategy_not_available_falling_back",
                requested=strategy.value,
                fallback="recursive",
            )
            strategy = ChunkingStrategy.RECURSIVE

        chunker = self._chunkers[strategy]
        return chunker.chunk(document)

    def list_strategies(self) -> list[str]:
        """List available chunking strategies."""
        return [s.value for s in self._chunkers.keys()]
