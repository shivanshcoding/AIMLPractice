#!/usr/bin/env python3
"""Generate evaluation datasets."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.datasets import EvaluationDataset

dataset = EvaluationDataset()
samples = EvaluationDataset.create_sample_dataset()
dataset.save(samples, "default")
print("Generated default evaluation dataset.")
