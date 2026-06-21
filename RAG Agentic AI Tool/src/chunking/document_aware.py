"""
Document-Aware Chunking Strategy.

Analyzes document structure (headings, sections, tables, lists)
and creates chunks that respect structural boundaries.
Never splits mid-table, mid-list, or mid-section.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from src.core.interfaces import BaseChunker
from src.core.models import Chunk, ChunkingStrategy, Document
from src.chunking.base import BaseChunkerMixin
from src.chunking.recursive import RecursiveChunker

logger = structlog.get_logger(__name__)


class DocumentAwareChunker(BaseChunker, BaseChunkerMixin):
    """
    Structure-aware chunking that respects document boundaries.

    Detects headings, tables, code blocks, and lists.
    Creates chunks at natural structural breaks.
    """

    def __init__(
        self,
        respect_headings: bool = True,
        respect_paragraphs: bool = True,
        respect_lists: bool = True,
        respect_tables: bool = True,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        fallback_strategy: str = "recursive",
        **kwargs,
    ) -> None:
        self._respect_headings = respect_headings
        self._respect_paragraphs = respect_paragraphs
        self._respect_lists = respect_lists
        self._respect_tables = respect_tables
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size
        # Fallback for chunks that are too large
        self._fallback = RecursiveChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=100,
        )

    @property
    def strategy_name(self) -> str:
        return "document_aware"

    def _detect_sections(self, text: str) -> list[dict[str, Any]]:
        """
        Detect structural sections in the document.

        Returns a list of sections with:
          - content: text content
          - title: section heading (if any)
          - type: heading | paragraph | table | list | code
          - level: heading level (1-6)
        """
        sections: list[dict[str, Any]] = []
        lines = text.split("\n")
        current_section: list[str] = []
        current_title = ""
        current_type = "paragraph"
        current_level = 0
        in_table = False
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Detect code blocks
            if stripped.startswith("```"):
                if in_code_block:
                    current_section.append(line)
                    sections.append({
                        "content": "\n".join(current_section),
                        "title": current_title,
                        "type": "code",
                        "level": current_level,
                    })
                    current_section = []
                    current_title = ""
                    current_type = "paragraph"
                    in_code_block = False
                else:
                    # Save current section before code block
                    if current_section:
                        sections.append({
                            "content": "\n".join(current_section),
                            "title": current_title,
                            "type": current_type,
                            "level": current_level,
                        })
                    current_section = [line]
                    current_type = "code"
                    in_code_block = True
                continue

            if in_code_block:
                current_section.append(line)
                continue

            # Detect markdown headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if heading_match and self._respect_headings:
                # Save current section
                if current_section:
                    sections.append({
                        "content": "\n".join(current_section),
                        "title": current_title,
                        "type": current_type,
                        "level": current_level,
                    })
                current_section = [line]
                current_title = heading_match.group(2)
                current_level = len(heading_match.group(1))
                current_type = "heading"
                continue

            # Detect tables
            if stripped.startswith("|") and self._respect_tables:
                if not in_table:
                    if current_section and current_type != "table":
                        sections.append({
                            "content": "\n".join(current_section),
                            "title": current_title,
                            "type": current_type,
                            "level": current_level,
                        })
                        current_section = []
                    in_table = True
                    current_type = "table"
                current_section.append(line)
                continue
            elif in_table:
                in_table = False
                sections.append({
                    "content": "\n".join(current_section),
                    "title": current_title,
                    "type": "table",
                    "level": current_level,
                })
                current_section = [line]
                current_type = "paragraph"
                continue

            # Detect lists
            list_match = re.match(r'^[\s]*[-*+•]\s|^\s*\d+\.\s', stripped)
            if list_match and self._respect_lists:
                if current_type != "list":
                    if current_section:
                        sections.append({
                            "content": "\n".join(current_section),
                            "title": current_title,
                            "type": current_type,
                            "level": current_level,
                        })
                        current_section = []
                    current_type = "list"
                current_section.append(line)
                continue

            # Paragraph break on empty lines
            if not stripped and self._respect_paragraphs:
                if current_section:
                    sections.append({
                        "content": "\n".join(current_section),
                        "title": current_title,
                        "type": current_type,
                        "level": current_level,
                    })
                    current_section = []
                    current_type = "paragraph"
                continue

            # Regular content
            if current_type == "list" and not list_match:
                sections.append({
                    "content": "\n".join(current_section),
                    "title": current_title,
                    "type": "list",
                    "level": current_level,
                })
                current_section = []
                current_type = "paragraph"

            current_section.append(line)

        # Handle remaining content
        if current_section:
            sections.append({
                "content": "\n".join(current_section),
                "title": current_title,
                "type": current_type,
                "level": current_level,
            })

        return [s for s in sections if s["content"].strip()]

    def _build_section_hierarchy(self, sections: list[dict[str, Any]]) -> list[str]:
        """Build a hierarchy path from section titles and levels."""
        hierarchy: list[str] = []
        for section in sections:
            if section["type"] == "heading":
                level = section["level"]
                # Trim hierarchy to current level
                hierarchy = hierarchy[:level - 1]
                hierarchy.append(section["title"])
        return hierarchy

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            metadata=document.metadata,
        )

    def chunk_text(self, text: str, document_id: str = "", **kwargs) -> list[Chunk]:
        metadata = kwargs.get("metadata", None)
        sections = self._detect_sections(text)

        if not sections:
            return self._fallback.chunk_text(text, document_id, **kwargs)

        chunks: list[Chunk] = []
        chunk_index = 0
        current_hierarchy: list[str] = []

        for section in sections:
            content = section["content"].strip()

            # Update hierarchy tracking
            if section["type"] == "heading":
                level = section["level"]
                current_hierarchy = current_hierarchy[:level - 1]
                current_hierarchy.append(section["title"])

            if not content:
                continue

            # If section is too large, use fallback splitting
            if len(content) > self._max_chunk_size:
                sub_chunks = self._fallback.chunk_text(
                    text=content, document_id=document_id, metadata=metadata
                )
                for sub in sub_chunks:
                    sub.chunking_strategy = ChunkingStrategy.DOCUMENT_AWARE
                    sub.chunk_index = chunk_index
                    sub.section_title = section.get("title", "")
                    sub.section_hierarchy = list(current_hierarchy)
                    chunks.append(sub)
                    chunk_index += 1
                continue

            # If section is too small, merge with next
            if len(content) < self._min_chunk_size:
                # Try to merge with next section in the list
                if chunks and len(chunks[-1].content) + len(content) <= self._max_chunk_size:
                    chunks[-1].content += "\n\n" + content
                    chunks[-1].token_count = self.count_tokens(chunks[-1].content)
                    continue

            start_char = text.find(content)
            chunks.append(
                self.create_chunk(
                    content=content,
                    document_id=document_id,
                    strategy=ChunkingStrategy.DOCUMENT_AWARE,
                    chunk_index=chunk_index,
                    start_char=max(0, start_char),
                    end_char=max(0, start_char) + len(content),
                    metadata=metadata,
                    section_title=section.get("title", ""),
                    section_hierarchy=list(current_hierarchy),
                )
            )
            chunk_index += 1

        logger.info(
            "document_aware_chunking_complete",
            sections_detected=len(sections),
            chunks_created=len(chunks),
        )
        return chunks if chunks else self._fallback.chunk_text(text, document_id, **kwargs)
