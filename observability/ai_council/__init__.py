"""
AI Council

Multi-agent AI system for daily summaries with zero hallucination guarantee.

Architecture:
- 7 Analyst Models: Diverse perspectives (GPT-4, Claude, Gemini, Grok, Llama, DeepSeek, Opus)
- 1 Judge Model: Synthesizes verified insights (Claude Opus)
- Number Verifier: Ensures no hallucinated numbers

Usage:
    from observability.ai_council import CouncilManager

    council = CouncilManager(api_keys={...})
    summary = await council.generate_daily_summary(date='2025-11-06')
"""

from observability.ai_council.council_manager import CouncilManager, CouncilSummary, AnalystReport
from observability.ai_council.number_verifier import NumberVerifier
from observability.ai_council.judge import JudgeModel

__all__ = [
    'CouncilManager',
    'CouncilSummary',
    'AnalystReport',
    'NumberVerifier',
    'JudgeModel'
]
