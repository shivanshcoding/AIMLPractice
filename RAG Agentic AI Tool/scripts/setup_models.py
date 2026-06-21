#!/usr/bin/env python3
"""
Setup Models script.
Downloads HuggingFace models locally for offline inference.
"""

from huggingface_hub import snapshot_download

models = [
    "BAAI/bge-m3",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-reranker-v2-m3"
]

for model in models:
    print(f"Downloading {model}...")
    snapshot_download(repo_id=model)
print("All models downloaded successfully.")
