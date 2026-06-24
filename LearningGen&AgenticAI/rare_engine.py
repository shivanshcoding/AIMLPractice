#!/usr/bin/env python3
"""RARE: Research-Grade Agentic Retrieval Engine - Script Backup"""
# This file is auto-generated from the notebook. Run rare_engine.ipynb on Kaggle.

# ====================================================================
# # RARE: Research-Grade Agentic Retrieval Engine
# 
# A self-contained Kaggle notebook implementing a configurable, benchmarkable, and explainable retrieval engine for Agentic AI systems.
# 
# **Capabilities:**
# - Multi-format document ingestion (PDF, DOCX, CSV, TXT, Markdown)
# - 5 chunking strategies with ensemble support
# - 4 retrieval methods with weighted fusion
# - Query optimization (MultiQuery, HyDE, StepBack, Expansion)
# - Ensemble reranking (BGE + CrossEncoder)
# - Context compression (Redundancy removal, Contextual, LLM)
# - Self-correcting retrieval with confidence-based retry loops
# - LangGraph workflow orchestration with conditional routing
# - Manual mode (full config control) and Agent mode (autonomous strategy selection)
# - DeepEval evaluation with Ollama
# - Benchmarking with CSV/leaderboard output
# - LangSmith observability (optional)
# ====================================================================

# ====================================================================
# ## Section 1: Environment Setup
# Install all required packages. This cell only needs to run once per Kaggle session.
# ====================================================================

import subprocess, sys, os

packages = [
    "langchain", "langchain-community", "langchain-experimental", "langchain-ollama",
    "langgraph",
    "qdrant-client",
    "rank_bm25",
    "pymupdf", "python-docx",
    "FlagEmbedding", "sentence-transformers",
    "deepeval",
    "langsmith",
    "tabulate",
]

print("Installing packages...")
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("All packages installed.")

# ====================================================================
# ## Section 2: Ollama Setup
# Install, start, and configure Ollama with health checks and model pull utilities.
# ====================================================================

import urllib.request
import subprocess
import time
import os
import shutil

def install_ollama():
    """Install Ollama on Linux (Kaggle environment)."""
    if shutil.which("ollama"):
        print("[OLLAMA] Already installed.")
        return True
    print("[OLLAMA] Installing...")
    ret = os.system("curl -fsSL https://ollama.com/install.sh | sh")
    if ret == 0:
        print("[OLLAMA] Installation complete.")
        return True
    print("[OLLAMA] Installation failed.")
    return False

def start_ollama():
    """Start Ollama server in background."""
    print("[OLLAMA] Starting server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for attempt in range(1, 16):
        try:
            resp = urllib.request.urlopen("http://localhost:11434/", timeout=2)
            if resp.getcode() == 200:
                print(f"[OLLAMA] Server ready (attempt {attempt}).")
                return True
        except Exception:
            print(f"[OLLAMA] Waiting... (attempt {attempt}/15)")
            time.sleep(2)
    print("[OLLAMA] Server failed to start.")
    return False

def pull_model(model_name):
    """Pull an Ollama model with status output."""
    print(f"[OLLAMA] Pulling model: {model_name}")
    result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OLLAMA] Model ready: {model_name}")
        return True
    print(f"[OLLAMA] Failed to pull {model_name}: {result.stderr[:200]}")
    return False

def check_ollama_health():
    """Verify Ollama is running and responsive."""
    try:
        resp = urllib.request.urlopen("http://localhost:11434/", timeout=5)
        return resp.getcode() == 200
    except Exception:
        return False

# Run setup
install_ollama()
start_ollama()

# ====================================================================
# ## Section 3: Configuration
# 
# All system behavior is controlled from this cell. Change values here to completely reconfigure
# the retrieval pipeline without modifying any source code.
# 
# **Modes:**
# - `agent_mode: False` -- Manual mode. Uses the exact configuration specified below.
# - `agent_mode: True` -- Agent mode. LLM autonomously selects strategies per query.
# 
# **Chunker/Retriever/Reranker weights** must sum to approximately 1.0.
# ====================================================================

# ---------------------------------------------------------------------------
# OLLAMA MODEL CONFIGURATION
# ---------------------------------------------------------------------------
OLLAMA_CONFIG = {
    "generation_model": "llama3",
    "embedding_model": "nomic-embed-text",
    "query_optimizer_model": "llama3",
    "evaluation_model": "llama3",
}

# ---------------------------------------------------------------------------
# PIPELINE CONFIGURATION
# ---------------------------------------------------------------------------
CONFIG = {
    "agent_mode": False,

    "chunkers": {
        "semantic": 0.5,
        "parent_child": 0.3,
        "recursive": 0.2,
    },

    "retrievers": {
        "dense": 0.5,
        "bm25": 0.3,
        "metadata": 0.2,
    },

    "rerankers": {
        "bge": 0.7,
        "cross_encoder": 0.3,
    },

    "query_optimizers": [
        "multi_query",
        "hyde",
        "step_back",
    ],

    "compression": "contextual",

    "top_k": 15,
    "confidence_threshold": 0.85,
    "max_retries": 3,
}

# ---------------------------------------------------------------------------
# DOCUMENT PATHS
# ---------------------------------------------------------------------------
DOCUMENTS = [
    # Paths will be auto-populated by sample document generator below.
    # Replace with your own document paths as needed.
]

# ---------------------------------------------------------------------------
# LANGSMITH CONFIGURATION (auto-detected)
# ---------------------------------------------------------------------------
LANGSMITH_CONFIG = {
    "project": "RARE",
    "enabled": False,
}

# ---------------------------------------------------------------------------
# RERANKER MODEL CONFIGURATION
# ---------------------------------------------------------------------------
RERANKER_CONFIG = {
    "bge_model": "BAAI/bge-reranker-v2-m3",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

# Pull configured Ollama models
for _model in set(OLLAMA_CONFIG.values()):
    pull_model(_model)

print("\n--- CONFIGURATION LOADED ---")
print(f"Agent Mode       : {CONFIG['agent_mode']}")
print(f"Generation Model : {OLLAMA_CONFIG['generation_model']}")
print(f"Embedding Model  : {OLLAMA_CONFIG['embedding_model']}")
print(f"Chunkers         : {list(CONFIG['chunkers'].keys())}")
print(f"Retrievers       : {list(CONFIG['retrievers'].keys())}")
print(f"Rerankers        : {list(CONFIG['rerankers'].keys())}")
print(f"Optimizers       : {CONFIG['query_optimizers']}")
print(f"Compression      : {CONFIG['compression']}")
print(f"Top K            : {CONFIG['top_k']}")
print(f"Confidence Thresh: {CONFIG['confidence_threshold']}")
print("--- END ---")

# ====================================================================
# ## Section 4: LangSmith Observability (Optional)
# 
# Configures LangSmith tracing if the API key is available via Kaggle Secrets.
# The system never fails due to missing LangSmith credentials.
# ====================================================================

import os

def configure_langsmith():
    """Configure LangSmith. Silently disables if unavailable."""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        api_key = secrets.get_secret("LANGSMITH_API_KEY")
        if api_key and api_key.strip():
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_CONFIG["project"]
            LANGSMITH_CONFIG["enabled"] = True
            print("--- LANGSMITH ---")
            print(f"Status  : Enabled")
            print(f"Project : {LANGSMITH_CONFIG['project']}")
            print("--- END ---")
            return True
    except Exception:
        pass
    LANGSMITH_CONFIG["enabled"] = False
    print("--- LANGSMITH ---")
    print("Status  : Disabled")
    print("--- END ---")
    return False

configure_langsmith()

# ====================================================================
# ## Section 5: Core Imports
# ====================================================================

import time
import hashlib
import json
import re
import uuid
import warnings
from typing import List, Dict, Optional, Any, Literal, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# LangChain / LangGraph
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# Vector store
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# BM25
from rank_bm25 import BM25Okapi

print("[IMPORTS] All core imports successful.")

@dataclass
class ScoredDocument:
    """A document with a relevance score and provenance metadata."""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_retriever: str = ""

    def to_langchain(self) -> Document:
        meta = {**self.metadata, "score": self.score, "source_retriever": self.source_retriever}
        return Document(page_content=self.content, metadata=meta)

@dataclass
class PipelineTrace:
    """Telemetry container for a single query execution."""
    query: str = ""
    query_type: str = ""
    chunkers_used: List[str] = field(default_factory=list)
    retrievers_used: List[str] = field(default_factory=list)
    optimizers_used: List[str] = field(default_factory=list)
    rerankers_used: List[str] = field(default_factory=list)
    compression_strategy: str = ""
    confidence_score: float = 0.0
    coverage_score: float = 0.0
    context_quality: float = 0.0
    docs_retrieved: int = 0
    docs_after_rerank: int = 0
    docs_final: int = 0
    retry_count: int = 0
    node_latencies: Dict[str, float] = field(default_factory=dict)
    total_latency: float = 0.0

print("[DATACLASSES] Defined ScoredDocument, PipelineTrace.")

# ====================================================================
# ## Section 6: Sample Document Generation
# Creates synthetic sample documents for demonstration and testing.
# ====================================================================

import csv

SAMPLE_DIR = "/kaggle/working/sample_docs"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- Sample TXT ---
SAMPLE_TXT = """Retrieval-Augmented Generation (RAG) is an AI framework that enhances large language
model outputs by incorporating external knowledge retrieval. RAG systems first retrieve relevant
documents from a knowledge base, then use those documents as context for generating responses.

Key components of a RAG pipeline include: document ingestion, text chunking, embedding generation,
vector storage, retrieval, reranking, and response generation. Each component can be individually
optimized to improve overall system performance.

Dense retrieval uses embedding similarity to find semantically relevant passages. BM25 retrieval
uses term frequency and inverse document frequency for keyword-based matching. Hybrid retrieval
combines both approaches for improved recall.

Chunking strategies significantly impact retrieval quality. Fixed-size chunking is simple but may
split sentences mid-thought. Semantic chunking preserves meaning by splitting at natural boundaries.
Parent-child chunking maintains hierarchical context by linking small retrieval units to larger
parent documents.

Reranking is a critical post-retrieval step that reorders documents using cross-encoder models.
Cross-encoders jointly encode the query and document, producing more accurate relevance scores
than bi-encoder approaches, at the cost of higher computational expense.

Query optimization techniques include multi-query generation, where the original query is
rephrased into multiple variants to improve recall. HyDE (Hypothetical Document Embeddings)
generates a hypothetical answer and uses it as a search query. Step-back prompting creates
a more abstract version of the query to capture broader context.
"""
txt_path = os.path.join(SAMPLE_DIR, "rag_overview.txt")
with open(txt_path, "w") as f:
    f.write(SAMPLE_TXT)

# --- Sample Markdown ---
SAMPLE_MD = """# Vector Database Architecture

## Overview
Vector databases are specialized storage systems designed for high-dimensional embedding vectors.
They enable fast approximate nearest neighbor (ANN) search, which is fundamental to modern
retrieval systems.

## Indexing Strategies

### HNSW (Hierarchical Navigable Small World)
HNSW constructs a multi-layer graph where each node represents a vector. The top layers contain
fewer nodes for fast coarse navigation, while bottom layers contain all nodes for precise search.
HNSW offers excellent query performance with logarithmic complexity.

### IVF (Inverted File Index)
IVF partitions the vector space into clusters using k-means. During search, only a subset of
clusters are scanned, trading recall for speed. IVF-PQ combines clustering with product
quantization for memory-efficient storage.

## Distance Metrics
- **Cosine Similarity**: Measures angle between vectors. Best for normalized embeddings.
- **Euclidean Distance**: Measures straight-line distance. Sensitive to magnitude.
- **Dot Product**: Fastest computation. Requires normalized vectors for meaningful results.

## Production Considerations
Scaling vector databases requires careful attention to index build time, memory consumption,
and query latency. Sharding across multiple nodes enables horizontal scaling. Replication
provides fault tolerance and read throughput.
"""
md_path = os.path.join(SAMPLE_DIR, "vector_db_arch.md")
with open(md_path, "w") as f:
    f.write(SAMPLE_MD)

# --- Sample CSV ---
csv_path = os.path.join(SAMPLE_DIR, "eval_questions.csv")
csv_rows = [
    ["question", "category", "difficulty"],
    ["What is RAG?", "conceptual", "easy"],
    ["How does HNSW indexing work?", "technical", "medium"],
    ["Compare dense and BM25 retrieval", "comparative", "medium"],
    ["What are the production considerations for vector databases?", "procedural", "hard"],
    ["Explain the difference between cosine similarity and euclidean distance", "conceptual", "medium"],
    ["What is parent-child chunking?", "technical", "easy"],
    ["How does HyDE query optimization work?", "technical", "medium"],
    ["What is cross-encoder reranking?", "technical", "medium"],
]
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

# --- Sample PDF (using PyMuPDF) ---
pdf_path = os.path.join(SAMPLE_DIR, "chunking_strategies.pdf")
try:
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    text_page1 = """Chunking Strategies for RAG Systems

1. Fixed-Size Chunking
Fixed-size chunking splits text into equal-length segments with optional overlap.
While simple to implement, it risks splitting sentences and paragraphs at
arbitrary points, potentially degrading retrieval quality.

Parameters: chunk_size (characters), chunk_overlap (characters)
Typical values: chunk_size=512, chunk_overlap=50

2. Recursive Character Splitting
This strategy attempts to split on natural boundaries in order of preference:
double newlines, single newlines, periods, spaces. It preserves paragraph
structure better than fixed-size chunking while maintaining size limits.

3. Semantic Chunking
Semantic chunking uses embedding similarity to detect topic boundaries. It
computes embeddings for each sentence, then identifies breakpoints where the
cosine similarity between consecutive sentences drops below a threshold. This
produces chunks that are semantically coherent."""
    p = fitz.Point(72, 72)
    page.insert_text(p, text_page1, fontsize=11)

    page2 = doc.new_page(width=612, height=792)
    text_page2 = """4. Parent-Child Chunking
Parent-child chunking creates a two-level hierarchy. Large parent chunks
(1000-2000 characters) provide broad context, while small child chunks
(200-400 characters) serve as retrieval units. When a child chunk is
retrieved, its parent is returned to provide richer context.

5. Document-Aware Chunking
This strategy adapts chunking behavior based on document type:
- PDF: Respects page boundaries
- Markdown: Splits by header hierarchy
- CSV: Each row becomes a separate chunk
- Code: Splits by function/class boundaries

Ensemble Chunking
Multiple chunking strategies can be combined in an ensemble. Each chunker
produces its own set of chunks, which are stored in separate vector
collections. During retrieval, results from all collections are fused
using Reciprocal Rank Fusion (RRF) with configurable weights."""
    page2.insert_text(fitz.Point(72, 72), text_page2, fontsize=11)
    doc.save(pdf_path)
    doc.close()
except Exception as e:
    print(f"[SAMPLE] PDF generation skipped: {e}")
    pdf_path = None

# --- Sample DOCX ---
docx_path = os.path.join(SAMPLE_DIR, "evaluation_methods.docx")
try:
    from docx import Document as DocxDocument
    ddoc = DocxDocument()
    ddoc.add_heading("RAG Evaluation Methods", level=1)
    ddoc.add_heading("Faithfulness", level=2)
    ddoc.add_paragraph(
        "Faithfulness measures whether the generated answer is grounded in the retrieved "
        "context. A faithful answer does not contain claims that cannot be attributed to "
        "the provided documents. This metric is critical for preventing hallucinations."
    )
    ddoc.add_heading("Context Precision", level=2)
    ddoc.add_paragraph(
        "Context precision evaluates whether the relevant documents are ranked higher in "
        "the retrieved set. High precision means the most useful documents appear first, "
        "reducing noise in the generation context window."
    )
    ddoc.add_heading("Context Recall", level=2)
    ddoc.add_paragraph(
        "Context recall measures whether all the necessary information to answer the query "
        "is present in the retrieved documents. Low recall indicates missing information "
        "that may lead to incomplete answers."
    )
    ddoc.add_heading("Answer Relevancy", level=2)
    ddoc.add_paragraph(
        "Answer relevancy assesses whether the generated response directly addresses the "
        "user query. An answer may be faithful to the context but irrelevant to the "
        "actual question being asked."
    )
    ddoc.add_heading("Hallucination Detection", level=2)
    ddoc.add_paragraph(
        "Hallucination detection identifies claims in the generated answer that are not "
        "supported by the retrieved context. This is measured by decomposing the answer "
        "into individual claims and checking each against the source documents."
    )
    ddoc.save(docx_path)
except Exception as e:
    print(f"[SAMPLE] DOCX generation skipped: {e}")
    docx_path = None

# Populate DOCUMENTS list
DOCUMENTS.clear()
for p in [txt_path, md_path, csv_path, pdf_path, docx_path]:
    if p and os.path.exists(p):
        DOCUMENTS.append(p)

print(f"[SAMPLE] Generated {len(DOCUMENTS)} sample documents in {SAMPLE_DIR}")
for d in DOCUMENTS:
    print(f"  - {os.path.basename(d)}")

# ====================================================================
# ## Section 7: Document Ingestion
# 
# Automatically detects document type, loads content, and extracts metadata.
# ====================================================================

class DocumentLoader:
    """Load and parse documents from multiple formats with metadata extraction."""

    SUPPORTED = {".pdf", ".docx", ".csv", ".txt", ".md"}

    def load(self, paths: List[str]) -> List[Document]:
        all_docs = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext not in self.SUPPORTED:
                print(f"[LOADER] Skipping unsupported format: {path}")
                continue
            try:
                loader = {
                    ".pdf": self._load_pdf,
                    ".docx": self._load_docx,
                    ".csv": self._load_csv,
                    ".txt": self._load_txt,
                    ".md": self._load_markdown,
                }[ext]
                docs = loader(path)
                for doc in docs:
                    doc.metadata.update(self._base_metadata(doc, path, ext))
                all_docs.extend(docs)
                print(f"[LOADER] {os.path.basename(path)}: {len(docs)} document(s)")
            except Exception as e:
                print(f"[LOADER] Error loading {path}: {e}")
        return all_docs

    def _base_metadata(self, doc: Document, path: str, ext: str) -> dict:
        return {
            "source": os.path.basename(path),
            "source_path": path,
            "doc_type": ext.lstrip("."),
            "word_count": len(doc.page_content.split()),
            "char_count": len(doc.page_content),
        }

    def _load_pdf(self, path: str) -> List[Document]:
        import fitz
        docs = []
        pdf = fitz.open(path)
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
        pdf.close()
        return docs

    def _load_docx(self, path: str) -> List[Document]:
        from docx import Document as DDoc
        ddoc = DDoc(path)
        paragraphs = [p.text.strip() for p in ddoc.paragraphs if p.text.strip()]
        # Group consecutive paragraphs into sections by headings
        sections, current = [], []
        for para in ddoc.paragraphs:
            if para.style.name.startswith("Heading") and current:
                sections.append("\n".join(current))
                current = []
            if para.text.strip():
                current.append(para.text.strip())
        if current:
            sections.append("\n".join(current))
        if not sections:
            sections = ["\n".join(paragraphs)]
        return [Document(page_content=s, metadata={"section": i + 1}) for i, s in enumerate(sections)]

    def _load_csv(self, path: str) -> List[Document]:
        df = pd.read_csv(path)
        docs = []
        for i, row in df.iterrows():
            content = " | ".join(f"{col}: {val}" for col, val in row.items())
            meta = {col: str(val) for col, val in row.items()}
            meta["row_index"] = i
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def _load_txt(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [Document(page_content=text)]
        return [Document(page_content=p, metadata={"paragraph": i + 1})
                for i, p in enumerate(paragraphs)]

    def _load_markdown(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        sections = re.split(r'\n(?=#{1,3}\s)', text)
        docs = []
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            header_match = re.match(r'^(#{1,3})\s+(.*)', section)
            header = header_match.group(2) if header_match else f"section_{i}"
            docs.append(Document(page_content=section, metadata={"section_header": header}))
        return docs if docs else [Document(page_content=text)]

# --- Run Ingestion ---
loader = DocumentLoader()
raw_documents = loader.load(DOCUMENTS)

print(f"\n--- INGESTION SUMMARY ---")
print(f"Documents Loaded : {len(raw_documents)}")
type_counts = defaultdict(int)
for d in raw_documents:
    type_counts[d.metadata.get("doc_type", "unknown")] += 1
for t, c in type_counts.items():
    print(f"  {t.upper():>8}: {c}")
print(f"Total Words      : {sum(d.metadata.get('word_count', 0) for d in raw_documents)}")
print("--- END ---")

# ====================================================================
# ## Section 8: Advanced Chunking
# 
# Five chunking strategies with ensemble support. Each chunker tags output with
# `chunker_type` metadata. Ensemble mode stores chunks from each strategy separately.
# ====================================================================

class FixedChunker:
    """Fixed-size chunking with configurable size and overlap."""
    name = "fixed"
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for c in chunks:
            c.metadata["chunker_type"] = self.name
        return chunks

class RecursiveChunker:
    """Recursive splitting with natural language separators."""
    name = "recursive"
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
    def chunk(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for c in chunks:
            c.metadata["chunker_type"] = self.name
        return chunks

class SemanticChunker:
    """Embedding-based semantic boundary detection."""
    name = "semantic"
    def __init__(self):
        self._fallback = RecursiveChunker()
        try:
            from langchain_experimental.text_splitter import SemanticChunker as SC
            self.splitter = SC(
                OllamaEmbeddings(model=OLLAMA_CONFIG["embedding_model"]),
                breakpoint_threshold_type="percentile",
            )
            self._available = True
        except Exception as e:
            print(f"[CHUNKER] SemanticChunker unavailable, using fallback: {e}")
            self._available = False

    def chunk(self, documents: List[Document]) -> List[Document]:
        if not self._available:
            chunks = self._fallback.chunk(documents)
            for c in chunks:
                c.metadata["chunker_type"] = self.name
            return chunks
        try:
            chunks = self.splitter.split_documents(documents)
            for c in chunks:
                c.metadata["chunker_type"] = self.name
            return chunks
        except Exception:
            chunks = self._fallback.chunk(documents)
            for c in chunks:
                c.metadata["chunker_type"] = self.name
            return chunks

class ParentChildChunker:
    """Two-level hierarchy: large parents for context, small children for retrieval."""
    name = "parent_child"
    def __init__(self, parent_size=1500, child_size=300, child_overlap=50):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=0
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap
        )
        self.parent_store = {}  # parent_id -> parent_content

    def chunk(self, documents: List[Document]) -> List[Document]:
        self.parent_store.clear()
        all_children = []
        parent_chunks = self.parent_splitter.split_documents(documents)
        for i, parent in enumerate(parent_chunks):
            parent_id = hashlib.md5(parent.page_content[:200].encode()).hexdigest()[:12]
            self.parent_store[parent_id] = parent.page_content
            children = self.child_splitter.split_documents([parent])
            for child in children:
                child.metadata["chunker_type"] = self.name
                child.metadata["parent_id"] = parent_id
                child.metadata["parent_index"] = i
            all_children.extend(children)
        return all_children

class DocumentAwareChunker:
    """Adapts chunking strategy based on document type."""
    name = "document_aware"
    def __init__(self):
        self.recursive = RecursiveChunker()

    def chunk(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            doc_type = doc.metadata.get("doc_type", "txt")
            if doc_type == "csv":
                doc.metadata["chunker_type"] = self.name
                all_chunks.append(doc)
            elif doc_type == "md":
                # Already split by headers in loader; just tag
                doc.metadata["chunker_type"] = self.name
                all_chunks.append(doc)
            else:
                chunks = self.recursive.chunk([doc])
                for c in chunks:
                    c.metadata["chunker_type"] = self.name
                all_chunks.extend(chunks)
        return all_chunks

CHUNKER_REGISTRY = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "parent_child": ParentChildChunker,
    "document_aware": DocumentAwareChunker,
}

print("[CHUNKERS] Registered:", list(CHUNKER_REGISTRY.keys()))

class ChunkEnsemble:
    """Orchestrates multiple chunkers. Produces separate chunk sets per strategy."""

    def __init__(self, chunker_config: dict):
        self.config = chunker_config  # {"semantic": 0.5, "recursive": 0.3, ...}
        self.chunkers = {}
        for name in chunker_config:
            if name in CHUNKER_REGISTRY:
                self.chunkers[name] = CHUNKER_REGISTRY[name]()
            else:
                print(f"[ENSEMBLE] Unknown chunker: {name}")
        self.parent_stores = {}  # name -> parent_store (for parent_child)

    def chunk(self, documents: List[Document]) -> Dict[str, List[Document]]:
        results = {}
        for name, chunker in self.chunkers.items():
            t0 = time.time()
            chunks = chunker.chunk(documents)
            elapsed = time.time() - t0
            results[name] = chunks
            if hasattr(chunker, "parent_store"):
                self.parent_stores[name] = chunker.parent_store
            print(f"[CHUNK] {name:>15}: {len(chunks):>5} chunks ({elapsed:.2f}s)")
        return results

# --- Run Chunking ---
chunk_ensemble = ChunkEnsemble(CONFIG["chunkers"])
chunked_docs = chunk_ensemble.chunk(raw_documents)

total_chunks = sum(len(v) for v in chunked_docs.values())
print(f"\n--- CHUNKING SUMMARY ---")
print(f"Total Chunk Sets : {len(chunked_docs)}")
print(f"Total Chunks     : {total_chunks}")
for name, chunks in chunked_docs.items():
    avg_len = np.mean([len(c.page_content) for c in chunks]) if chunks else 0
    print(f"  {name:>15}: {len(chunks)} chunks, avg {avg_len:.0f} chars")
print("--- END ---")

# ====================================================================
# ## Section 9: Vector Store (Qdrant Local)
# 
# Stores embeddings in Qdrant running in local mode. Each chunker strategy gets its own collection.
# ====================================================================

class VectorStoreManager:
    """Manages Qdrant collections for document storage and retrieval."""

    def __init__(self, path="/kaggle/working/qdrant_data"):
        self.client = QdrantClient(path=path)
        self.embedder = OllamaEmbeddings(model=OLLAMA_CONFIG["embedding_model"])
        self._dim = None
        self.collections = {}

    def _get_dim(self) -> int:
        if self._dim is None:
            test = self.embedder.embed_query("dimension test")
            self._dim = len(test)
        return self._dim

    def create_collection(self, name: str, chunks: List[Document], batch_size: int = 32):
        """Embed and store chunks in a named collection."""
        collection_name = f"rare_{name}"
        dim = self._get_dim()

        # Recreate collection
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # Batch embed and upsert
        points = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.page_content for c in batch]
            try:
                embeddings = self.embedder.embed_documents(texts)
            except Exception as e:
                print(f"[VECTOR] Embedding error at batch {i}: {e}")
                continue
            for j, (emb, chunk) in enumerate(zip(embeddings, batch)):
                point_id = str(uuid.uuid4())
                payload = {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata,
                    "chunk_index": i + j,
                }
                points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        if points:
            # Upsert in batches
            for i in range(0, len(points), 100):
                self.client.upsert(collection_name=collection_name, points=points[i:i+100])

        self.collections[name] = collection_name
        print(f"[VECTOR] Collection '{collection_name}': {len(points)} vectors (dim={dim})")
        return collection_name

    def search(self, collection_name: str, query: str, top_k: int = 10) -> List[ScoredDocument]:
        """Search a collection by query embedding."""
        try:
            query_emb = self.embedder.embed_query(query)
        except Exception as e:
            print(f"[VECTOR] Query embedding error: {e}")
            return []
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_emb,
            limit=top_k,
        ).points
        docs = []
        for r in results:
            docs.append(ScoredDocument(
                content=r.payload.get("text", ""),
                score=r.score,
                metadata=r.payload.get("metadata", {}),
                source_retriever="dense",
            ))
        return docs

# --- Build Vector Stores ---
vector_manager = VectorStoreManager()
for chunker_name, chunks in chunked_docs.items():
    if chunks:
        vector_manager.create_collection(chunker_name, chunks)

print(f"\n--- VECTOR STORE SUMMARY ---")
print(f"Collections: {list(vector_manager.collections.keys())}")
print("--- END ---")

class BM25Index:
    """In-memory BM25 index for keyword-based retrieval."""

    def __init__(self):
        self.indices = {}    # name -> BM25Okapi
        self.doc_store = {}  # name -> List[Document]

    def build(self, name: str, chunks: List[Document]):
        tokenized = [c.page_content.lower().split() for c in chunks]
        if not tokenized:
            return
        self.indices[name] = BM25Okapi(tokenized)
        self.doc_store[name] = chunks
        print(f"[BM25] Index '{name}': {len(chunks)} documents")

    def search(self, name: str, query: str, top_k: int = 10) -> List[ScoredDocument]:
        if name not in self.indices:
            return []
        tokens = query.lower().split()
        scores = self.indices[name].get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.doc_store[name][idx]
                results.append(ScoredDocument(
                    content=doc.page_content,
                    score=float(scores[idx]),
                    metadata=doc.metadata,
                    source_retriever="bm25",
                ))
        return results

# --- Build BM25 Indices ---
bm25_index = BM25Index()
for chunker_name, chunks in chunked_docs.items():
    if chunks:
        bm25_index.build(chunker_name, chunks)

print("[BM25] All indices built.")

# ====================================================================
# ## Section 10: Retrieval Engine
# 
# Four retrieval strategies with weighted ensemble fusion via Reciprocal Rank Fusion (RRF).
# ====================================================================

class DenseRetriever:
    """Embedding-based semantic retrieval via Qdrant."""
    name = "dense"
    def __init__(self, vector_manager: VectorStoreManager):
        self.vm = vector_manager
    def retrieve(self, query: str, top_k: int, collections: List[str] = None) -> List[ScoredDocument]:
        cols = collections or list(self.vm.collections.values())
        all_results = []
        for col in cols:
            all_results.extend(self.vm.search(col, query, top_k))
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

class BM25Retriever:
    """Keyword-based BM25 retrieval."""
    name = "bm25"
    def __init__(self, bm25_idx: BM25Index):
        self.bm25 = bm25_idx
    def retrieve(self, query: str, top_k: int, index_names: List[str] = None) -> List[ScoredDocument]:
        names = index_names or list(self.bm25.indices.keys())
        all_results = []
        for name in names:
            all_results.extend(self.bm25.search(name, query, top_k))
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

class MetadataRetriever:
    """Retrieves documents matching metadata filters extracted from the query."""
    name = "metadata"
    def __init__(self, vector_manager: VectorStoreManager):
        self.vm = vector_manager
        self.llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0)

    def _extract_keywords(self, query: str) -> List[str]:
        try:
            prompt = ChatPromptTemplate.from_template(
                "Extract 2-4 key technical terms from this query. "
                "Return ONLY a comma-separated list, nothing else.\nQuery: {query}"
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})
            return [kw.strip().lower() for kw in result.split(",") if kw.strip()]
        except Exception:
            return query.lower().split()[:4]

    def retrieve(self, query: str, top_k: int, **kwargs) -> List[ScoredDocument]:
        keywords = self._extract_keywords(query)
        # Search across all BM25 indices using extracted keywords as a refined query
        refined_query = " ".join(keywords)
        all_results = []
        for name in bm25_index.indices:
            all_results.extend(bm25_index.search(name, refined_query, top_k))
        all_results.sort(key=lambda x: x.score, reverse=True)
        for r in all_results:
            r.source_retriever = "metadata"
        return all_results[:top_k]

class ParentChildRetriever:
    """Retrieves child chunks then returns their parent documents for richer context."""
    name = "parent_child"
    def __init__(self, vector_manager: VectorStoreManager, parent_stores: dict):
        self.vm = vector_manager
        self.parent_stores = parent_stores

    def retrieve(self, query: str, top_k: int, **kwargs) -> List[ScoredDocument]:
        # Search parent_child collection if it exists
        pc_col = self.vm.collections.get("parent_child")
        if not pc_col:
            return []
        children = self.vm.search(pc_col, query, top_k * 2)
        # Map children to parents
        seen_parents = set()
        results = []
        for child in children:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                for store in self.parent_stores.values():
                    if parent_id in store:
                        results.append(ScoredDocument(
                            content=store[parent_id],
                            score=child.score,
                            metadata={**child.metadata, "retrieval_type": "parent"},
                            source_retriever="parent_child",
                        ))
                        seen_parents.add(parent_id)
                        break
            if len(results) >= top_k:
                break
        return results[:top_k]

RETRIEVER_REGISTRY = {}

def build_retrievers():
    global RETRIEVER_REGISTRY
    RETRIEVER_REGISTRY = {
        "dense": DenseRetriever(vector_manager),
        "bm25": BM25Retriever(bm25_index),
        "metadata": MetadataRetriever(vector_manager),
        "parent_child": ParentChildRetriever(vector_manager, chunk_ensemble.parent_stores),
    }

build_retrievers()
print("[RETRIEVERS] Registered:", list(RETRIEVER_REGISTRY.keys()))

class RetrievalEnsemble:
    """Weighted multi-retriever fusion using Reciprocal Rank Fusion."""

    def __init__(self, retriever_config: dict, k: int = 60):
        self.config = retriever_config
        self.k = k  # RRF constant

    def retrieve(self, queries: List[str], top_k: int) -> List[ScoredDocument]:
        """Run all configured retrievers on all queries, fuse with RRF."""
        # Collect ranked lists: retriever_name -> list of ScoredDocuments
        retriever_results = defaultdict(list)

        for rname, weight in self.config.items():
            if weight <= 0 or rname not in RETRIEVER_REGISTRY:
                continue
            retriever = RETRIEVER_REGISTRY[rname]
            for query in queries:
                try:
                    results = retriever.retrieve(query, top_k)
                    retriever_results[rname].extend(results)
                except Exception as e:
                    print(f"[RETRIEVAL] {rname} error: {e}")

        # RRF fusion
        doc_scores = defaultdict(float)
        doc_map = {}
        doc_sources = defaultdict(set)

        for rname, docs in retriever_results.items():
            weight = self.config.get(rname, 1.0)
            # Deduplicate within retriever and assign ranks
            seen = set()
            ranked = []
            for d in docs:
                key = hashlib.md5(d.content[:200].encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    ranked.append((key, d))

            for rank, (key, d) in enumerate(ranked):
                rrf_score = weight / (self.k + rank + 1)
                doc_scores[key] += rrf_score
                doc_map[key] = d
                doc_sources[key].add(rname)

        # Sort by fused score
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            d = doc_map[key]
            results.append(ScoredDocument(
                content=d.content,
                score=doc_scores[key],
                metadata={**d.metadata, "fusion_sources": list(doc_sources[key])},
                source_retriever="ensemble",
            ))
        return results

print("[ENSEMBLE] RetrievalEnsemble ready.")

# ====================================================================
# ## Section 11: Query Optimization
# 
# Multiple query optimization strategies can be enabled simultaneously.
# ====================================================================

class MultiQueryOptimizer:
    """Generates multiple query reformulations for improved recall."""
    name = "multi_query"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["query_optimizer_model"], temperature=0.7)
        self.prompt = ChatPromptTemplate.from_template(
            "Generate 3 alternative search queries for the following question. "
            "Each should approach the topic from a different angle.\n"
            "Return ONLY the 3 queries, one per line, no numbering.\n\n"
            "Original: {query}"
        )
    def optimize(self, query: str) -> List[str]:
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({"query": query})
            alternatives = [q.strip() for q in result.strip().split("\n") if q.strip()]
            return [query] + alternatives[:3]
        except Exception:
            return [query]

class QueryExpansionOptimizer:
    """Expands query with related terms and synonyms."""
    name = "query_expansion"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["query_optimizer_model"], temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(
            "Expand this search query by adding relevant technical terms, "
            "synonyms, and related concepts. Return a single expanded query.\n\n"
            "Original: {query}"
        )
    def optimize(self, query: str) -> List[str]:
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            expanded = chain.invoke({"query": query}).strip()
            return [query, expanded]
        except Exception:
            return [query]

class HyDEOptimizer:
    """Generates a hypothetical answer to use as retrieval query."""
    name = "hyde"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["query_optimizer_model"], temperature=0.5)
        self.prompt = ChatPromptTemplate.from_template(
            "Write a short, factual paragraph that would be the ideal answer to this question. "
            "This will be used as a search query. Do not explain, just write the answer.\n\n"
            "Question: {query}"
        )
    def optimize(self, query: str) -> List[str]:
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            hypothetical = chain.invoke({"query": query}).strip()
            return [query, hypothetical]
        except Exception:
            return [query]

class StepBackOptimizer:
    """Creates a more abstract version of the query for broader retrieval."""
    name = "step_back"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["query_optimizer_model"], temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(
            "Given this specific question, generate a more general, abstract question "
            "that would help retrieve broader background information.\n"
            "Return ONLY the abstract question.\n\n"
            "Specific: {query}"
        )
    def optimize(self, query: str) -> List[str]:
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            abstract = chain.invoke({"query": query}).strip()
            return [query, abstract]
        except Exception:
            return [query]

OPTIMIZER_REGISTRY = {
    "multi_query": MultiQueryOptimizer,
    "query_expansion": QueryExpansionOptimizer,
    "hyde": HyDEOptimizer,
    "step_back": StepBackOptimizer,
}

print("[OPTIMIZERS] Registered:", list(OPTIMIZER_REGISTRY.keys()))

# ====================================================================
# ## Section 12: Reranking
# 
# BGE and CrossEncoder rerankers with weighted ensemble support.
# ====================================================================

class BGEReranker:
    """BAAI BGE reranker using FlagEmbedding."""
    name = "bge"
    def __init__(self):
        self._available = False
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(RERANKER_CONFIG["bge_model"], use_fp16=True)
            self._available = True
            print(f"[RERANKER] BGE loaded: {RERANKER_CONFIG['bge_model']}")
        except Exception as e:
            print(f"[RERANKER] BGE unavailable: {e}")

    def rerank(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        if not self._available or not documents:
            return documents
        pairs = [[query, d.content] for d in documents]
        try:
            scores = self.model.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            for d, s in zip(documents, scores):
                d.score = float(s)
                d.source_retriever = f"{d.source_retriever}+bge"
            documents.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            print(f"[RERANKER] BGE scoring error: {e}")
        return documents

class CrossEncoderReranker:
    """Sentence-transformers CrossEncoder reranker."""
    name = "cross_encoder"
    def __init__(self):
        self._available = False
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(RERANKER_CONFIG["cross_encoder_model"])
            self._available = True
            print(f"[RERANKER] CrossEncoder loaded: {RERANKER_CONFIG['cross_encoder_model']}")
        except Exception as e:
            print(f"[RERANKER] CrossEncoder unavailable: {e}")

    def rerank(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        if not self._available or not documents:
            return documents
        pairs = [(query, d.content) for d in documents]
        try:
            scores = self.model.predict(pairs)
            for d, s in zip(documents, scores):
                d.score = float(s)
                d.source_retriever = f"{d.source_retriever}+ce"
            documents.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            print(f"[RERANKER] CrossEncoder scoring error: {e}")
        return documents

class EnsembleReranker:
    """Weighted combination of multiple rerankers."""
    def __init__(self, reranker_config: dict):
        self.config = reranker_config
        self.rerankers = {}
        if "bge" in reranker_config:
            self.rerankers["bge"] = BGEReranker()
        if "cross_encoder" in reranker_config:
            self.rerankers["cross_encoder"] = CrossEncoderReranker()

    def rerank(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        if not documents:
            return documents

        # Get scores from each reranker
        score_sets = {}
        for rname, reranker in self.rerankers.items():
            docs_copy = [ScoredDocument(d.content, d.score, dict(d.metadata), d.source_retriever) for d in documents]
            reranked = reranker.rerank(query, docs_copy)
            scores = {hashlib.md5(d.content[:200].encode()).hexdigest(): d.score for d in reranked}
            # Normalize scores to [0, 1]
            if scores:
                min_s, max_s = min(scores.values()), max(scores.values())
                rng = max_s - min_s if max_s > min_s else 1.0
                scores = {k: (v - min_s) / rng for k, v in scores.items()}
            score_sets[rname] = scores

        # Weighted combination
        for doc in documents:
            key = hashlib.md5(doc.content[:200].encode()).hexdigest()
            combined = 0.0
            for rname, scores in score_sets.items():
                weight = self.config.get(rname, 0.5)
                combined += weight * scores.get(key, 0.0)
            doc.score = combined

        documents.sort(key=lambda x: x.score, reverse=True)
        return documents

# Initialize rerankers
ensemble_reranker = EnsembleReranker(CONFIG["rerankers"])
print("[RERANKERS] Ensemble ready.")

# ====================================================================
# ## Section 13: Context Compression
# 
# Three compression strategies to reduce context size while preserving relevance.
# ====================================================================

class RedundancyRemovalCompressor:
    """Removes near-duplicate documents using content hashing and similarity."""
    name = "redundancy_removal"
    def compress(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        if not documents:
            return documents
        unique = []
        seen_hashes = set()
        for doc in documents:
            h = hashlib.md5(doc.content.strip().lower()[:300].encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(doc)
        return unique

class ContextualCompressor:
    """LLM extracts only query-relevant content from each document."""
    name = "contextual"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "Extract ONLY the sentences from the following text that are directly relevant "
            "to answering the query. If nothing is relevant, respond with 'NOT_RELEVANT'.\n\n"
            "Query: {query}\n\nText: {text}"
        )
    def compress(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        chain = self.prompt | self.llm | StrOutputParser()
        compressed = []
        for doc in documents[:10]:  # Limit to avoid excessive LLM calls
            try:
                result = chain.invoke({"query": query, "text": doc.content[:1500]})
                if result.strip() and result.strip() != "NOT_RELEVANT":
                    compressed.append(ScoredDocument(
                        content=result.strip(),
                        score=doc.score,
                        metadata={**doc.metadata, "compressed": True},
                        source_retriever=doc.source_retriever,
                    ))
            except Exception:
                compressed.append(doc)
        return compressed if compressed else documents[:5]

class LLMCompressor:
    """LLM summarizes combined context into a concise passage."""
    name = "llm"
    def __init__(self):
        self.llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "Condense the following retrieved passages into a single concise summary "
            "that preserves all information relevant to the query.\n\n"
            "Query: {query}\n\nPassages:\n{passages}"
        )
    def compress(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        passages = "\n---\n".join(d.content[:500] for d in documents[:8])
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"query": query, "passages": passages})
            return [ScoredDocument(
                content=summary.strip(),
                score=1.0,
                metadata={"compressed": True, "strategy": "llm_summary"},
                source_retriever="llm_compressor",
            )]
        except Exception:
            return documents[:5]

COMPRESSOR_REGISTRY = {
    "redundancy_removal": RedundancyRemovalCompressor,
    "contextual": ContextualCompressor,
    "llm": LLMCompressor,
}

print("[COMPRESSORS] Registered:", list(COMPRESSOR_REGISTRY.keys()))

class ExplainabilityLogger:
    """Professional engineering-grade execution logs."""

    def log_query_analysis(self, trace: PipelineTrace, detected_entities: List[str] = None):
        print("\n--- QUERY ANALYSIS ---")
        print(f"Query              : {trace.query}")
        print(f"Detected Query Type: {trace.query_type}")
        print(f"Entities           : {detected_entities or []}")
        print(f"Selected Chunkers  : {trace.chunkers_used}")
        print(f"Selected Retrievers: {trace.retrievers_used}")
        print(f"Selected Optimizers: {trace.optimizers_used}")
        print(f"Selected Rerankers : {trace.rerankers_used}")
        print(f"Selected Compress. : {trace.compression_strategy}")
        print("--- END ---")

    def log_retrieval(self, trace: PipelineTrace, retriever_contributions: dict = None):
        print("\n--- RETRIEVAL EXECUTION ---")
        print(f"Documents Retrieved     : {trace.docs_retrieved}")
        print(f"Fusion Strategy         : Reciprocal Rank Fusion (RRF)")
        print(f"Retriever Contributions : {retriever_contributions or {}}")
        print(f"Latency                 : {trace.node_latencies.get('retrieval', 0):.3f}s")
        print("--- END ---")

    def log_reranking(self, trace: PipelineTrace, before: int = 0, avg_score: float = 0.0):
        print("\n--- RERANKING ---")
        print(f"Documents Before : {before}")
        print(f"Documents After  : {trace.docs_after_rerank}")
        print(f"Average Score    : {avg_score:.4f}")
        print(f"Latency          : {trace.node_latencies.get('reranking', 0):.3f}s")
        print("--- END ---")

    def log_confidence(self, trace: PipelineTrace):
        print("\n--- CONFIDENCE ANALYSIS ---")
        print(f"Coverage Score        : {trace.coverage_score:.4f}")
        print(f"Context Quality Score : {trace.context_quality:.4f}")
        print(f"Retrieval Confidence  : {trace.confidence_score:.4f}")
        print(f"Retry Required        : {trace.confidence_score < CONFIG['confidence_threshold']}")
        print(f"Retry Count           : {trace.retry_count}")
        print("--- END ---")

    def log_summary(self, trace: PipelineTrace):
        print("\n--- RARE EXECUTION SUMMARY ---")
        print(f"Query Type    : {trace.query_type}")
        print(f"Chunkers      : {trace.chunkers_used}")
        print(f"Retrievers    : {trace.retrievers_used}")
        print(f"Optimizers    : {trace.optimizers_used}")
        print(f"Rerankers     : {trace.rerankers_used}")
        print(f"Compression   : {trace.compression_strategy}")
        print(f"Confidence    : {trace.confidence_score:.4f}")
        print(f"Total Latency : {trace.total_latency:.3f}s")
        print(f"Documents Used: {trace.docs_final}")
        for node, lat in trace.node_latencies.items():
            print(f"  {node:>25}: {lat:.3f}s")
        print("--- END ---")

    def log_langsmith(self, trace: PipelineTrace):
        print("\n--- LANGSMITH TRACE ---")
        print(f"Project      : {LANGSMITH_CONFIG['project']}")
        print(f"Trace Status : {'Enabled' if LANGSMITH_CONFIG['enabled'] else 'Disabled'}")
        print(f"Query Type   : {trace.query_type}")
        print(f"Nodes Exec.  : {len(trace.node_latencies)}")
        print(f"Total Latency: {trace.total_latency:.3f}s")
        print("--- END ---")

explainability = ExplainabilityLogger()
print("[LOGGER] ExplainabilityLogger ready.")

# ====================================================================
# ## Section 14: LangGraph Workflow
# 
# The complete retrieval pipeline orchestrated as a LangGraph state machine with
# conditional routing for self-correcting retrieval.
# 
# ```
# Query Analysis -> Strategy Selection -> Query Optimization -> Retrieval
# -> Reranking -> Confidence Evaluation --(low confidence)--> Retry Loop
#                                       --(high confidence)--> Compression -> Response
# ```
# ====================================================================

from typing import TypedDict, Annotated

class RAREState(TypedDict):
    """Complete state for the RARE LangGraph workflow."""
    # Input
    query: str
    config_override: dict

    # Query Analysis
    query_type: str
    detected_entities: list

    # Strategy
    selected_chunkers: dict
    selected_retrievers: dict
    selected_optimizers: list
    selected_rerankers: dict
    selected_compression: str
    selected_top_k: int

    # Execution Data
    optimized_queries: list
    retrieved_documents: list  # List of dicts (serializable ScoredDocuments)
    reranked_documents: list
    compressed_documents: list

    # Confidence
    confidence_score: float
    coverage_score: float
    context_quality: float
    retry_count: int
    max_retries: int

    # Output
    final_answer: str

    # Telemetry
    trace: dict
    node_latencies: dict


def _scored_to_dict(sd: ScoredDocument) -> dict:
    return {"content": sd.content, "score": sd.score, "metadata": sd.metadata, "source_retriever": sd.source_retriever}

def _dict_to_scored(d: dict) -> ScoredDocument:
    return ScoredDocument(d["content"], d["score"], d.get("metadata", {}), d.get("source_retriever", ""))


# ---- NODE 1: Query Analysis ----
def query_analysis_node(state: RAREState) -> dict:
    t0 = time.time()
    query = state["query"]
    llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0)

    # Classify query type
    classify_prompt = ChatPromptTemplate.from_template(
        "Classify this query into exactly one category: factual, analytical, procedural, comparative.\n"
        "Return ONLY the category name, nothing else.\n\nQuery: {query}"
    )
    try:
        chain = classify_prompt | llm | StrOutputParser()
        query_type = chain.invoke({"query": query}).strip().lower()
        if query_type not in {"factual", "analytical", "procedural", "comparative"}:
            query_type = "factual"
    except Exception:
        query_type = "factual"

    # Extract entities
    entity_prompt = ChatPromptTemplate.from_template(
        "Extract key technical entities/terms from this query. "
        "Return a comma-separated list.\n\nQuery: {query}"
    )
    try:
        chain = entity_prompt | llm | StrOutputParser()
        entities_str = chain.invoke({"query": query})
        entities = [e.strip() for e in entities_str.split(",") if e.strip()]
    except Exception:
        entities = []

    latency = time.time() - t0
    return {
        "query_type": query_type,
        "detected_entities": entities,
        "node_latencies": {**state.get("node_latencies", {}), "query_analysis": latency},
    }


# ---- NODE 2: Strategy Selection ----
def strategy_selection_node(state: RAREState) -> dict:
    t0 = time.time()
    cfg = state.get("config_override") or CONFIG

    if cfg.get("agent_mode", False):
        # Agent autonomously selects strategies
        llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0.3)
        agent_prompt = ChatPromptTemplate.from_template(
            "You are a retrieval strategy expert. Select the optimal retrieval configuration.\n\n"
            "Query: {query}\nQuery Type: {query_type}\n"
            "Available chunkers: semantic, parent_child, recursive, fixed, document_aware\n"
            "Available retrievers: dense, bm25, metadata, parent_child\n"
            "Available optimizers: multi_query, hyde, step_back, query_expansion\n"
            "Available rerankers: bge, cross_encoder\n"
            "Available compression: redundancy_removal, contextual, llm\n\n"
            "Respond in valid JSON only:\n"
            '{{"chunkers": {{"name": weight}}, "retrievers": {{"name": weight}}, '
            '"rerankers": {{"name": weight}}, "optimizers": ["name"], '
            '"compression": "name", "top_k": int}}'
        )
        try:
            chain = agent_prompt | llm | StrOutputParser()
            raw = chain.invoke({"query": state["query"], "query_type": state["query_type"]})
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                strategy = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except Exception:
            # Fallback to defaults
            strategy = {
                "chunkers": cfg["chunkers"],
                "retrievers": cfg["retrievers"],
                "rerankers": cfg["rerankers"],
                "optimizers": cfg["query_optimizers"],
                "compression": cfg["compression"],
                "top_k": cfg["top_k"],
            }
    else:
        strategy = {
            "chunkers": cfg["chunkers"],
            "retrievers": cfg["retrievers"],
            "rerankers": cfg["rerankers"],
            "optimizers": cfg["query_optimizers"],
            "compression": cfg["compression"],
            "top_k": cfg["top_k"],
        }

    latency = time.time() - t0
    return {
        "selected_chunkers": strategy.get("chunkers", cfg["chunkers"]),
        "selected_retrievers": strategy.get("retrievers", cfg["retrievers"]),
        "selected_optimizers": strategy.get("optimizers", cfg["query_optimizers"]),
        "selected_rerankers": strategy.get("rerankers", cfg["rerankers"]),
        "selected_compression": strategy.get("compression", cfg["compression"]),
        "selected_top_k": strategy.get("top_k", cfg["top_k"]),
        "node_latencies": {**state.get("node_latencies", {}), "strategy_selection": latency},
    }


# ---- NODE 3: Query Optimization ----
def query_optimization_node(state: RAREState) -> dict:
    t0 = time.time()
    query = state["query"]
    optimizer_names = state.get("selected_optimizers", [])

    all_queries = {query}  # Use set to avoid duplicates
    for opt_name in optimizer_names:
        if opt_name in OPTIMIZER_REGISTRY:
            try:
                optimizer = OPTIMIZER_REGISTRY[opt_name]()
                expanded = optimizer.optimize(query)
                all_queries.update(expanded)
            except Exception as e:
                print(f"[OPTIMIZER] {opt_name} error: {e}")

    latency = time.time() - t0
    return {
        "optimized_queries": list(all_queries),
        "node_latencies": {**state.get("node_latencies", {}), "query_optimization": latency},
    }


# ---- NODE 4: Retrieval ----
def retrieval_node(state: RAREState) -> dict:
    t0 = time.time()
    queries = state.get("optimized_queries", [state["query"]])
    top_k = state.get("selected_top_k", CONFIG["top_k"])
    retriever_config = state.get("selected_retrievers", CONFIG["retrievers"])

    ensemble = RetrievalEnsemble(retriever_config)
    results = ensemble.retrieve(queries, top_k)

    latency = time.time() - t0
    return {
        "retrieved_documents": [_scored_to_dict(r) for r in results],
        "node_latencies": {**state.get("node_latencies", {}), "retrieval": latency},
    }


# ---- NODE 5: Reranking ----
def reranking_node(state: RAREState) -> dict:
    t0 = time.time()
    docs = [_dict_to_scored(d) for d in state.get("retrieved_documents", [])]

    if docs and ensemble_reranker.rerankers:
        reranked = ensemble_reranker.rerank(state["query"], docs)
    else:
        reranked = docs

    latency = time.time() - t0
    return {
        "reranked_documents": [_scored_to_dict(r) for r in reranked],
        "node_latencies": {**state.get("node_latencies", {}), "reranking": latency},
    }


# ---- NODE 6: Confidence Evaluation ----
def confidence_evaluation_node(state: RAREState) -> dict:
    t0 = time.time()
    docs = [_dict_to_scored(d) for d in state.get("reranked_documents", [])]
    query = state["query"]

    # Coverage: what fraction of query terms appear in retrieved docs
    query_terms = set(query.lower().split())
    if docs and query_terms:
        doc_text = " ".join(d.content.lower() for d in docs[:10])
        covered = sum(1 for t in query_terms if t in doc_text)
        coverage = covered / len(query_terms)
    else:
        coverage = 0.0

    # Context quality: average normalized score of top docs
    if docs:
        top_scores = [d.score for d in docs[:5]]
        max_s = max(top_scores) if top_scores else 1.0
        quality = np.mean([s / max_s for s in top_scores]) if max_s > 0 else 0.0
    else:
        quality = 0.0

    confidence = 0.6 * coverage + 0.4 * float(quality)
    retry_count = state.get("retry_count", 0)

    latency = time.time() - t0
    return {
        "confidence_score": confidence,
        "coverage_score": coverage,
        "context_quality": float(quality),
        "retry_count": retry_count,
        "node_latencies": {**state.get("node_latencies", {}), "confidence_evaluation": latency},
    }


# ---- CONDITIONAL: Should Retry? ----
def should_retry(state: RAREState) -> Literal["retry", "continue"]:
    threshold = CONFIG.get("confidence_threshold", 0.85)
    max_r = state.get("max_retries", CONFIG.get("max_retries", 3))
    if state.get("confidence_score", 0) < threshold and state.get("retry_count", 0) < max_r:
        return "retry"
    return "continue"


# ---- NODE 7: Retry Adjustment ----
def retry_adjustment_node(state: RAREState) -> dict:
    """Adjust strategy for retry: increase top_k, add optimizers."""
    t0 = time.time()
    retry_count = state.get("retry_count", 0) + 1
    current_top_k = state.get("selected_top_k", CONFIG["top_k"])
    new_top_k = int(current_top_k * 1.5)

    optimizers = list(state.get("selected_optimizers", []))
    if "query_expansion" not in optimizers:
        optimizers.append("query_expansion")
    if retry_count >= 2 and "hyde" not in optimizers:
        optimizers.append("hyde")

    # Broaden retrievers on second retry
    retrievers = dict(state.get("selected_retrievers", CONFIG["retrievers"]))
    if retry_count >= 2:
        for r in ["dense", "bm25", "metadata"]:
            if r not in retrievers:
                retrievers[r] = 0.2

    latency = time.time() - t0
    print(f"[RETRY] Attempt {retry_count}: top_k={new_top_k}, optimizers={optimizers}")
    return {
        "retry_count": retry_count,
        "selected_top_k": new_top_k,
        "selected_optimizers": optimizers,
        "selected_retrievers": retrievers,
        "node_latencies": {**state.get("node_latencies", {}), f"retry_{retry_count}": latency},
    }


# ---- NODE 8: Compression ----
def compression_node(state: RAREState) -> dict:
    t0 = time.time()
    docs = [_dict_to_scored(d) for d in state.get("reranked_documents", [])]
    strategy_name = state.get("selected_compression", CONFIG["compression"])

    # Always apply redundancy removal first
    rr = RedundancyRemovalCompressor()
    docs = rr.compress(state["query"], docs)

    # Then apply selected compression
    if strategy_name in COMPRESSOR_REGISTRY and strategy_name != "redundancy_removal":
        try:
            compressor = COMPRESSOR_REGISTRY[strategy_name]()
            docs = compressor.compress(state["query"], docs)
        except Exception as e:
            print(f"[COMPRESSION] {strategy_name} error: {e}")

    latency = time.time() - t0
    return {
        "compressed_documents": [_scored_to_dict(d) for d in docs],
        "node_latencies": {**state.get("node_latencies", {}), "compression": latency},
    }


# ---- NODE 9: Response Generation ----
def response_generation_node(state: RAREState) -> dict:
    t0 = time.time()
    docs = state.get("compressed_documents", [])
    query = state["query"]

    context = "\n\n---\n\n".join(d["content"] for d in docs[:8])
    llm = ChatOllama(model=OLLAMA_CONFIG["generation_model"], temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based on the provided context. "
        "If the context does not contain enough information, say so.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    try:
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "query": query})
    except Exception as e:
        answer = f"Error generating response: {e}"

    latency = time.time() - t0
    return {
        "final_answer": answer.strip(),
        "node_latencies": {**state.get("node_latencies", {}), "response_generation": latency},
    }

print("[NODES] All LangGraph nodes defined.")

# ---- Build the LangGraph Workflow ----
workflow = StateGraph(RAREState)

# Add nodes
workflow.add_node("query_analysis", query_analysis_node)
workflow.add_node("strategy_selection", strategy_selection_node)
workflow.add_node("query_optimization", query_optimization_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("reranking", reranking_node)
workflow.add_node("confidence_evaluation", confidence_evaluation_node)
workflow.add_node("retry_adjustment", retry_adjustment_node)
workflow.add_node("compression", compression_node)
workflow.add_node("response_generation", response_generation_node)

# Set entry point
workflow.set_entry_point("query_analysis")

# Linear edges
workflow.add_edge("query_analysis", "strategy_selection")
workflow.add_edge("strategy_selection", "query_optimization")
workflow.add_edge("query_optimization", "retrieval")
workflow.add_edge("retrieval", "reranking")
workflow.add_edge("reranking", "confidence_evaluation")

# Conditional edge: retry or continue
workflow.add_conditional_edges(
    "confidence_evaluation",
    should_retry,
    {"retry": "retry_adjustment", "continue": "compression"},
)
workflow.add_edge("retry_adjustment", "query_optimization")

# Final edges
workflow.add_edge("compression", "response_generation")
workflow.add_edge("response_generation", END)

# Compile
rare_app = workflow.compile()
print("[LANGGRAPH] Workflow compiled successfully.")
print("[LANGGRAPH] Nodes:", [n for n in ["query_analysis", "strategy_selection",
    "query_optimization", "retrieval", "reranking", "confidence_evaluation",
    "retry_adjustment", "compression", "response_generation"]])

# ====================================================================
# ## Section 15: Query Interface
# 
# The main entry point for running queries through the RARE pipeline.
# ====================================================================

def query_rare(q: str, config_override: dict = None, verbose: bool = True) -> dict:
    """Execute a query through the full RARE pipeline.

    Args:
        q: The query string.
        config_override: Optional config dict to override global CONFIG.
        verbose: If True, print explainability logs.

    Returns:
        dict with keys: answer, trace, state
    """
    t_start = time.time()

    initial_state = {
        "query": q,
        "config_override": config_override or CONFIG,
        "query_type": "",
        "detected_entities": [],
        "selected_chunkers": {},
        "selected_retrievers": {},
        "selected_optimizers": [],
        "selected_rerankers": {},
        "selected_compression": "",
        "selected_top_k": CONFIG["top_k"],
        "optimized_queries": [],
        "retrieved_documents": [],
        "reranked_documents": [],
        "compressed_documents": [],
        "confidence_score": 0.0,
        "coverage_score": 0.0,
        "context_quality": 0.0,
        "retry_count": 0,
        "max_retries": CONFIG.get("max_retries", 3),
        "final_answer": "",
        "trace": {},
        "node_latencies": {},
    }

    # Run the graph
    result = rare_app.invoke(initial_state)
    total_latency = time.time() - t_start

    # Build trace
    trace = PipelineTrace(
        query=q,
        query_type=result.get("query_type", ""),
        chunkers_used=list(result.get("selected_chunkers", {}).keys()),
        retrievers_used=list(result.get("selected_retrievers", {}).keys()),
        optimizers_used=result.get("selected_optimizers", []),
        rerankers_used=list(result.get("selected_rerankers", {}).keys()),
        compression_strategy=result.get("selected_compression", ""),
        confidence_score=result.get("confidence_score", 0),
        coverage_score=result.get("coverage_score", 0),
        context_quality=result.get("context_quality", 0),
        docs_retrieved=len(result.get("retrieved_documents", [])),
        docs_after_rerank=len(result.get("reranked_documents", [])),
        docs_final=len(result.get("compressed_documents", [])),
        retry_count=result.get("retry_count", 0),
        node_latencies=result.get("node_latencies", {}),
        total_latency=total_latency,
    )

    if verbose:
        explainability.log_query_analysis(trace, result.get("detected_entities"))
        explainability.log_retrieval(trace)
        explainability.log_reranking(trace, before=trace.docs_retrieved,
            avg_score=np.mean([d["score"] for d in result.get("reranked_documents", [{}])]) if result.get("reranked_documents") else 0)
        explainability.log_confidence(trace)
        explainability.log_summary(trace)
        explainability.log_langsmith(trace)

    return {
        "answer": result.get("final_answer", ""),
        "trace": trace,
        "state": result,
    }

print("[INTERFACE] query_rare() ready.")

# ====================================================================
# ## Section 16: Evaluation (DeepEval)
# 
# Evaluates retrieval and generation quality using DeepEval with Ollama.
# ====================================================================

class RAREEvaluator:
    """Evaluation engine using DeepEval with local Ollama models."""

    def __init__(self):
        self.model = None
        try:
            from deepeval.models import OllamaModel
            self.model = OllamaModel(
                model=OLLAMA_CONFIG["evaluation_model"],
                base_url="http://localhost:11434",
            )
            print("[EVAL] DeepEval + Ollama configured.")
        except Exception as e:
            print(f"[EVAL] DeepEval setup error (will use fallback): {e}")

    def evaluate(self, test_cases: List[dict]) -> pd.DataFrame:
        """Run evaluation on a list of test cases.

        Each test case: {query, answer, context_list, ground_truth}
        Returns a DataFrame with metric scores.
        """
        results = []
        if self.model is None:
            print("[EVAL] No evaluation model available. Using heuristic fallback.")
            return self._fallback_evaluate(test_cases)

        try:
            from deepeval.test_case import LLMTestCase
            from deepeval.metrics import (
                FaithfulnessMetric,
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                HallucinationMetric,
            )
        except ImportError as e:
            print(f"[EVAL] DeepEval import error: {e}")
            return self._fallback_evaluate(test_cases)

        metrics = {
            "faithfulness": FaithfulnessMetric(model=self.model, threshold=0.5),
            "answer_relevancy": AnswerRelevancyMetric(model=self.model, threshold=0.5),
            "contextual_precision": ContextualPrecisionMetric(model=self.model, threshold=0.5),
            "contextual_recall": ContextualRecallMetric(model=self.model, threshold=0.5),
            "hallucination": HallucinationMetric(model=self.model, threshold=0.5),
        }

        for i, tc in enumerate(test_cases):
            case = LLMTestCase(
                input=tc["query"],
                actual_output=tc["answer"],
                retrieval_context=tc.get("context_list", []),
                expected_output=tc.get("ground_truth", ""),
            )
            row = {"query": tc["query"]}
            for mname, metric in metrics.items():
                try:
                    metric.measure(case)
                    row[mname] = metric.score
                except Exception as e:
                    row[mname] = None
                    print(f"[EVAL] {mname} error on case {i}: {e}")
            results.append(row)
            print(f"[EVAL] Case {i+1}/{len(test_cases)} complete.")

        return pd.DataFrame(results)

    def _fallback_evaluate(self, test_cases: List[dict]) -> pd.DataFrame:
        """Heuristic evaluation when DeepEval is unavailable."""
        results = []
        for tc in test_cases:
            answer = tc.get("answer", "").lower()
            context = " ".join(tc.get("context_list", [])).lower()
            query = tc.get("query", "").lower()

            # Simple heuristic metrics
            query_terms = set(query.split())
            answer_terms = set(answer.split())
            context_terms = set(context.split())

            relevancy = len(query_terms & answer_terms) / max(len(query_terms), 1)
            coverage = len(query_terms & context_terms) / max(len(query_terms), 1)

            results.append({
                "query": tc["query"],
                "answer_relevancy": min(relevancy * 1.5, 1.0),
                "context_coverage": min(coverage * 1.2, 1.0),
                "note": "heuristic_fallback",
            })
        return pd.DataFrame(results)

evaluator = RAREEvaluator()
print("[EVAL] Evaluator ready.")

def run_evaluation(queries: List[str] = None) -> pd.DataFrame:
    """Run evaluation on sample or provided queries."""
    if queries is None:
        queries = [
            "What is RAG?",
            "How does HNSW indexing work?",
            "What are the different chunking strategies?",
            "Compare dense and BM25 retrieval methods",
        ]

    test_cases = []
    for q in queries:
        print(f"\n[EVAL-RUN] Querying: {q}")
        result = query_rare(q, verbose=False)
        context_list = [d["content"] for d in result["state"].get("compressed_documents", [])]
        test_cases.append({
            "query": q,
            "answer": result["answer"],
            "context_list": context_list,
            "ground_truth": "",  # No ground truth for demo
        })

    df = evaluator.evaluate(test_cases)

    # Save results
    out_path = "/kaggle/working/evaluation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[EVAL] Results saved to {out_path}")
    print("\n--- EVALUATION RESULTS ---")
    print(df.to_string(index=False))
    print("--- END ---")
    return df

# Uncomment to run evaluation:
# eval_df = run_evaluation()

# ====================================================================
# ## Section 17: Benchmarking
# 
# Systematically benchmarks different retrieval configurations and produces comparison reports.
# ====================================================================

class BenchmarkRunner:
    """Benchmarks RARE configurations across chunkers, retrievers, rerankers, and optimizers."""

    def __init__(self, queries: List[str] = None):
        self.queries = queries or [
            "What is RAG?",
            "How does semantic chunking work?",
            "Compare HNSW and IVF indexing",
            "What is cross-encoder reranking?",
        ]
        self.results = []

    def _run_config(self, name: str, config: dict) -> dict:
        """Run all queries with a specific config, return aggregate metrics."""
        latencies = []
        confidences = []
        doc_counts = []

        for q in self.queries:
            try:
                result = query_rare(q, config_override=config, verbose=False)
                trace = result["trace"]
                latencies.append(trace.total_latency)
                confidences.append(trace.confidence_score)
                doc_counts.append(trace.docs_final)
            except Exception as e:
                print(f"[BENCH] Error with '{name}' on '{q[:40]}': {e}")
                latencies.append(0)
                confidences.append(0)
                doc_counts.append(0)

        return {
            "configuration": name,
            "avg_latency": np.mean(latencies),
            "avg_confidence": np.mean(confidences),
            "avg_docs_final": np.mean(doc_counts),
            "queries_run": len(self.queries),
        }

    def benchmark_configurations(self) -> pd.DataFrame:
        """Run predefined benchmark configurations."""
        configs = {
            "Dense Only": {**CONFIG, "retrievers": {"dense": 1.0}, "query_optimizers": []},
            "BM25 Only": {**CONFIG, "retrievers": {"bm25": 1.0}, "query_optimizers": []},
            "Hybrid (Dense+BM25)": {**CONFIG, "retrievers": {"dense": 0.6, "bm25": 0.4}, "query_optimizers": []},
            "Hybrid + MultiQuery": {**CONFIG, "retrievers": {"dense": 0.6, "bm25": 0.4}, "query_optimizers": ["multi_query"]},
            "Hybrid + HyDE": {**CONFIG, "retrievers": {"dense": 0.6, "bm25": 0.4}, "query_optimizers": ["hyde"]},
            "Full Pipeline": CONFIG,
        }

        print("[BENCH] Starting benchmark run...")
        for name, cfg in configs.items():
            print(f"\n[BENCH] Configuration: {name}")
            result = self._run_config(name, cfg)
            self.results.append(result)
            print(f"  Avg Latency   : {result['avg_latency']:.3f}s")
            print(f"  Avg Confidence: {result['avg_confidence']:.4f}")

        df = pd.DataFrame(self.results)
        return df

    def generate_reports(self, df: pd.DataFrame):
        """Generate benchmark output files."""
        # benchmark_results.csv
        df.to_csv("/kaggle/working/benchmark_results.csv", index=False)

        # leaderboard.csv (sorted by confidence)
        leaderboard = df.sort_values("avg_confidence", ascending=False).reset_index(drop=True)
        leaderboard.index += 1
        leaderboard.index.name = "rank"
        leaderboard.to_csv("/kaggle/working/leaderboard.csv")

        # comparison_report.md
        report = "# RARE Benchmark Report\n\n"
        report += f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "## Leaderboard\n\n"
        report += "| Rank | Configuration | Avg Latency (s) | Avg Confidence | Avg Docs |\n"
        report += "|------|--------------|-----------------|----------------|----------|\n"
        for i, row in leaderboard.iterrows():
            report += f"| {i} | {row['configuration']} | {row['avg_latency']:.3f} | {row['avg_confidence']:.4f} | {row['avg_docs_final']:.1f} |\n"
        report += "\n## Configuration Details\n\n"
        report += "Each configuration was tested against the same query set.\n"
        report += f"Queries: {len(self.queries)}\n"

        with open("/kaggle/working/comparison_report.md", "w") as f:
            f.write(report)

        print("\n--- BENCHMARK OUTPUTS ---")
        print("benchmark_results.csv  : /kaggle/working/benchmark_results.csv")
        print("leaderboard.csv        : /kaggle/working/leaderboard.csv")
        print("comparison_report.md   : /kaggle/working/comparison_report.md")
        print("--- END ---")

        print("\n--- LEADERBOARD ---")
        print(leaderboard.to_string())
        print("--- END ---")

# Uncomment to run benchmarks:
# bench = BenchmarkRunner()
# bench_df = bench.benchmark_configurations()
# bench.generate_reports(bench_df)

# ====================================================================
# ## Section 18: Interactive Querying
# 
# Run queries through the RARE pipeline with full explainability output.
# ====================================================================

# ---- Example 1: Manual Mode Query ----
print("=" * 70)
print("RARE ENGINE - MANUAL MODE QUERY")
print("=" * 70)

result = query_rare("What is RAG and how does retrieval work?")

print("\n" + "=" * 70)
print("GENERATED ANSWER:")
print("=" * 70)
print(result["answer"])

# ---- Example 2: Agent Mode Query ----
print("=" * 70)
print("RARE ENGINE - AGENT MODE QUERY")
print("=" * 70)

agent_config = {**CONFIG, "agent_mode": True}
result_agent = query_rare(
    "Compare semantic chunking and parent-child chunking strategies",
    config_override=agent_config,
)

print("\n" + "=" * 70)
print("GENERATED ANSWER:")
print("=" * 70)
print(result_agent["answer"])

# ---- Example 3: Run Evaluation ----
# Uncomment the line below to run full evaluation
# eval_df = run_evaluation()
print("[INFO] Uncomment the line above to run evaluation.")
print("[INFO] Results will be saved to /kaggle/working/evaluation_results.csv")

# ---- Example 4: Run Benchmarks ----
# Uncomment the lines below to run full benchmarks
# bench = BenchmarkRunner()
# bench_df = bench.benchmark_configurations()
# bench.generate_reports(bench_df)
print("[INFO] Uncomment the lines above to run benchmarks.")
print("[INFO] Results will be saved to /kaggle/working/benchmark_results.csv")
print("[INFO] Leaderboard will be saved to /kaggle/working/leaderboard.csv")

# ====================================================================
# ## End of Notebook
# 
# RARE: Research-Grade Agentic Retrieval Engine
# 
# To customize the pipeline, modify the `CONFIG` and `OLLAMA_CONFIG` dictionaries in Section 3.
# No source code changes are required to change system behavior.
# ====================================================================

