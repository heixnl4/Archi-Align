"""
评估模块：提供 LLM-as-a-Judge 自动化评测能力。
"""

from .llm_judge_eval import evaluate_single_case, run_evaluation

__all__ = [
    "evaluate_single_case",
    "run_evaluation",
]
