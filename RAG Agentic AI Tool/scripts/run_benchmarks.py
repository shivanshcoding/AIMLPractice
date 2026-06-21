#!/usr/bin/env python3
"""Run benchmarking suite."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmarking.runner import BenchmarkRunner

print("Running benchmarks...")
runner = BenchmarkRunner()
print("Done.")
