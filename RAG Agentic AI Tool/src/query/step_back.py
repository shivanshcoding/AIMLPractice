"""
Step-Back Prompting.

Generates a more abstract "step-back" question to retrieve broader
context before answering the specific query.
"""

from __future__ import annotations

import structlog

from src.core.interfaces import BaseLLM

logger = structlog.get_logger(__name__)

STEP_BACK_PROMPT = """You are an expert at generating step-back questions.
A step-back question is a more general or abstract version of the original question
that helps retrieve broader context needed to answer the specific question.

Original question: {query}

Generate ONE step-back question that captures the broader concept or principle
behind this question.

Step-back question:"""


class StepBackGenerator:
    """
    Step-Back Prompting.

    Generates abstract questions that retrieve broader foundational context.
    Best for analytical and procedural queries.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def generate_step_back_query(self, query: str) -> str:
        """
        Generate a step-back question.

        Args:
            query: The original specific query.

        Returns:
            A more abstract step-back question.
        """
        try:
            prompt = STEP_BACK_PROMPT.format(query=query)
            response = self._llm.generate(prompt, temperature=0.3, max_tokens=128)
            step_back = response.strip()

            # Clean up any prefixes
            if step_back.lower().startswith("step-back question:"):
                step_back = step_back[len("step-back question:"):].strip()

            logger.debug(
                "step_back_generated",
                original=query[:50],
                step_back=step_back[:50],
            )
            return step_back

        except Exception as e:
            logger.warning("step_back_failed", error=str(e))
            return query
