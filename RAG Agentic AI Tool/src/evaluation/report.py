"""
Evaluation Report Generator.

Generates evaluation reports in CSV, JSON, and Markdown formats.
Creates leaderboard tables comparing configurations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.core.models import BenchmarkResult, EvaluationResult

logger = structlog.get_logger(__name__)


class ReportGenerator:
    """Generates evaluation and benchmark reports."""

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_report(
        self,
        results: list[EvaluationResult],
        name: str = "evaluation",
    ) -> dict[str, str]:
        """Generate reports in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{name}_{timestamp}"

        paths = {}

        # JSON
        json_path = self._output_dir / f"{prefix}.json"
        with open(json_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)
        paths["json"] = str(json_path)

        # CSV
        csv_path = self._output_dir / f"{prefix}.csv"
        if results:
            headers = list(results[0].model_dump().keys())
            with open(csv_path, "w") as f:
                f.write(",".join(headers) + "\n")
                for r in results:
                    values = [str(v) for v in r.model_dump().values()]
                    f.write(",".join(values) + "\n")
        paths["csv"] = str(csv_path)

        # Markdown
        md_path = self._output_dir / f"{prefix}.md"
        md_content = self._build_markdown_report(results, name)
        with open(md_path, "w") as f:
            f.write(md_content)
        paths["markdown"] = str(md_path)

        logger.info("reports_generated", paths=paths)
        return paths

    def generate_benchmark_leaderboard(
        self,
        results: list[BenchmarkResult],
        name: str = "benchmark",
    ) -> dict[str, str]:
        """Generate a benchmark leaderboard."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{name}_leaderboard_{timestamp}"

        paths = {}

        # JSON
        json_path = self._output_dir / f"{prefix}.json"
        with open(json_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, default=str)
        paths["json"] = str(json_path)

        # Markdown leaderboard
        md_path = self._output_dir / f"{prefix}.md"
        md = self._build_leaderboard_markdown(results)
        with open(md_path, "w") as f:
            f.write(md)
        paths["markdown"] = str(md_path)

        return paths

    def _build_markdown_report(
        self, results: list[EvaluationResult], name: str
    ) -> str:
        """Build a markdown evaluation report."""
        lines = [
            f"# Evaluation Report: {name}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Metrics Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
        ]

        if results:
            r = results[0]
            if r.faithfulness is not None:
                lines.append(f"| Faithfulness | {r.faithfulness:.4f} |")
            if r.answer_relevancy is not None:
                lines.append(f"| Answer Relevancy | {r.answer_relevancy:.4f} |")
            if r.context_precision is not None:
                lines.append(f"| Context Precision | {r.context_precision:.4f} |")
            if r.context_recall is not None:
                lines.append(f"| Context Recall | {r.context_recall:.4f} |")
            if r.answer_correctness is not None:
                lines.append(f"| Answer Correctness | {r.answer_correctness:.4f} |")
            if r.mrr is not None:
                lines.append(f"| MRR | {r.mrr:.4f} |")
            if r.ndcg is not None:
                lines.append(f"| NDCG | {r.ndcg:.4f} |")
            for k, v in r.recall_at_k.items():
                lines.append(f"| Recall@{k} | {v:.4f} |")
            for k, v in r.precision_at_k.items():
                lines.append(f"| Precision@{k} | {v:.4f} |")

        return "\n".join(lines)

    def _build_leaderboard_markdown(self, results: list[BenchmarkResult]) -> str:
        """Build a markdown leaderboard table."""
        lines = [
            "# Benchmark Leaderboard",
            f"Generated: {datetime.now().isoformat()}",
            "",
        ]

        if not results:
            lines.append("No results available.")
            return "\n".join(lines)

        # Collect all metric names
        all_metrics: set[str] = set()
        for r in results:
            all_metrics.update(r.metrics.keys())
        metric_names = sorted(all_metrics)

        # Build table
        header = "| Config | " + " | ".join(metric_names) + " | p50 (ms) | p95 (ms) |"
        separator = "|" + "|".join(["--------"] * (len(metric_names) + 3)) + "|"

        lines.extend([header, separator])

        for r in results:
            row = f"| {r.config_name} |"
            for m in metric_names:
                val = r.metrics.get(m, 0.0)
                row += f" {val:.4f} |"
            row += f" {r.latency_p50_ms:.1f} | {r.latency_p95_ms:.1f} |"
            lines.append(row)

        return "\n".join(lines)
