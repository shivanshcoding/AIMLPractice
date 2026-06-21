# RAG Agentic AI Platform - Kaggle Research Edition

A modular, extensible Retrieval-Augmented Generation (RAG) research platform optimized for Kaggle notebook environments with local GPU inference (Tesla T4/P100).

## Features

- **Local GPU Inference**: Direct loading of `Qwen` via `transformers` with 4-bit `bitsandbytes` quantization.
- **Advanced Chunking**: Fixed, Recursive, Semantic, Parent-Child, and Document-Aware structure-based chunking with an auto-selector heuristic engine.
- **Robust Embeddings**: Multi-modal BGE-M3 (Dense + Sparse in a single pass) loaded locally on GPU.
- **Local Indexing**: Native disk-backed Qdrant Vector Store + BM25 Sparse Index, supporting Hybrid Fusion (RRF, Weighted).
- **Query Understanding Layer**: Intent classification, Query Expansion, Multi-Query generation, HyDE, and Step-back prompting.
- **LangGraph Retrieval Agent**: Adaptive retrieval workflow with confidence evaluation, context compression, and self-correcting loops.
- **Reranking**: BGE Reranker v2 M3 (Cross-Encoder) executed locally.
- **Context Compression**: Redundancy Removal, Contextual Compression, LLM Summarization, and Context Packing.
- **Evaluation & Benchmarking**: RAGAS and DeepEval integrations, pure-Python IR metrics (NDCG, MRR, Recall@K), and automated leaderboards.
- **Observability**: Native LangSmith integration.

## Quickstart (Kaggle)

1. Upload this repository to Kaggle as a Dataset.
2. Create a new Notebook and attach the Dataset.
3. Select a **T4 x2** or **P100** GPU accelerator.
4. Run `notebooks/01_rag_research_pipeline.ipynb` to execute the end-to-end pipeline.

Alternatively, you can manually install the dependencies:
```bash
pip install -r requirements-kaggle.txt
```

## Architecture
See `src/` for core components. The platform is designed to execute locally without any external database servers or Docker containers.
