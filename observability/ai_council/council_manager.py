"""
AI Council Manager

Coordinates 8 analyst models + 1 judge to produce verified daily summaries.

Architecture:
- 8 Analysts: Each analyzes metrics independently (diverse reasoning)
- 1 Judge: Synthesizes verified claims into final summary
- Number Verifier: Prevents hallucination by checking every number

This ensures:
- No hallucinated numbers (4-layer verification)
- Diverse perspectives (8 different models)
- Synthesis quality (judge model combines insights)

Usage:
    from observability.ai_council.council_manager import CouncilManager

    council = CouncilManager(api_keys={
        'openai': 'sk-...',
        'anthropic': 'sk-...',
        'google': 'AIza...',
        'xai': '...',
        'groq': '...',
        'deepseek': '...',
        'perplexity': '...'
    })

    # Get daily summary
    summary = await council.generate_daily_summary(date='2025-11-06')

    print(summary['final_summary'])
    # "Engine learned X, improved by Y%, ready for Hamilton: Z"
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog
from pathlib import Path
import json

from observability.analytics.metrics_computer import MetricsComputer
from observability.ai_council.number_verifier import NumberVerifier

logger = structlog.get_logger(__name__)


@dataclass
class AnalystReport:
    """Individual analyst's report"""
    analyst_name: str
    model_name: str
    summary: str
    key_insights: List[str]
    numbers_cited: Dict[str, float]
    verified: bool
    verification_errors: List[str]


@dataclass
class CouncilSummary:
    """Final synthesized summary from judge"""
    date: str
    final_summary: str
    key_learnings: List[str]
    recommendations: List[str]
    hamilton_ready: bool
    analyst_reports: List[AnalystReport]
    verification_status: str
    total_analysts: int
    verified_analysts: int


class CouncilManager:
    """
    Manage AI Council for daily summaries.

    Coordinates:
    - 8 analyst models (diverse reasoning)
    - 1 judge model (synthesis)
    - Number verification (anti-hallucination)
    """

    def __init__(
        self,
        api_keys: Dict[str, str],
        cache_dir: str = "observability/data/ai_council_cache",
        enable_cache: bool = True
    ):
        """
        Initialize AI Council.

        Args:
            api_keys: Dict with keys: openai, anthropic, google, xai, groq, deepseek, perplexity
            cache_dir: Directory for caching summaries
            enable_cache: Whether to cache summaries (avoid re-running)
        """
        self.api_keys = api_keys
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = enable_cache

        # Initialize components
        self.metrics_computer = MetricsComputer()
        self.number_verifier = NumberVerifier()

        # Import analysts (lazy import to avoid import errors if APIs not available)
        self.analysts = self._initialize_analysts()
        self.judge = self._initialize_judge()

        logger.info(
            "council_manager_initialized",
            total_analysts=len(self.analysts),
            cache_enabled=enable_cache
        )

    def _initialize_analysts(self) -> List[Any]:
        """Initialize all analyst models"""
        from observability.ai_council.analysts.gpt4_analyst import GPT4Analyst
        from observability.ai_council.analysts.claude_sonnet_analyst import ClaudeSonnetAnalyst
        from observability.ai_council.analysts.gemini_analyst import GeminiAnalyst
        from observability.ai_council.analysts.grok_analyst import GrokAnalyst
        from observability.ai_council.analysts.llama_analyst import LlamaAnalyst
        from observability.ai_council.analysts.deepseek_analyst import DeepSeekAnalyst
        from observability.ai_council.analysts.claude_opus_analyst import ClaudeOpusAnalyst
        from observability.ai_council.analysts.perplexity_analyst import PerplexityAnalyst

        analysts = []

        # GPT-4-Turbo
        if 'openai' in self.api_keys:
            analysts.append(GPT4Analyst(api_key=self.api_keys['openai']))

        # Claude Sonnet
        if 'anthropic' in self.api_keys:
            analysts.append(ClaudeSonnetAnalyst(api_key=self.api_keys['anthropic']))

        # Gemini 1.5 Pro
        if 'google' in self.api_keys:
            analysts.append(GeminiAnalyst(api_key=self.api_keys['google']))

        # Grok 2
        if 'xai' in self.api_keys:
            analysts.append(GrokAnalyst(api_key=self.api_keys['xai']))

        # Llama 3 70B (via Groq)
        if 'groq' in self.api_keys:
            analysts.append(LlamaAnalyst(api_key=self.api_keys['groq']))

        # DeepSeek-R1
        if 'deepseek' in self.api_keys:
            analysts.append(DeepSeekAnalyst(api_key=self.api_keys['deepseek']))

        # Claude Opus (as analyst, not judge)
        if 'anthropic' in self.api_keys:
            analysts.append(ClaudeOpusAnalyst(api_key=self.api_keys['anthropic']))

        # Perplexity AI
        if 'perplexity' in self.api_keys:
            analysts.append(PerplexityAnalyst(api_key=self.api_keys['perplexity']))

        return analysts

    def _initialize_judge(self) -> Any:
        """Initialize judge model (Claude Opus)"""
        from observability.ai_council.judge import JudgeModel

        if 'anthropic' not in self.api_keys:
            raise ValueError("Anthropic API key required for judge model")

        return JudgeModel(api_key=self.api_keys['anthropic'])

    async def generate_daily_summary(
        self,
        date: str,
        force_refresh: bool = False
    ) -> CouncilSummary:
        """
        Generate daily summary using AI Council.

        Args:
            date: Date string (YYYY-MM-DD)
            force_refresh: Bypass cache and regenerate

        Returns:
            CouncilSummary with verified insights
        """
        logger.info("generating_daily_summary", date=date, force_refresh=force_refresh)

        # Check cache
        if not force_refresh and self.enable_cache:
            cached = self._load_from_cache(date)
            if cached:
                logger.info("summary_loaded_from_cache", date=date)
                return cached

        # Step 1: Pre-compute metrics (anti-hallucination layer)
        logger.info("computing_metrics_for_ai_council", date=date)
        metrics = self.metrics_computer.compute_daily_metrics(date)

        # Step 2: Run all analysts in parallel
        logger.info("running_analysts", count=len(self.analysts))
        analyst_tasks = [
            self._run_analyst(analyst, date, metrics)
            for analyst in self.analysts
        ]
        analyst_reports = await asyncio.gather(*analyst_tasks)

        # Step 3: Verify all numbers (anti-hallucination)
        logger.info("verifying_analyst_reports")
        verified_reports = []
        for report in analyst_reports:
            verified = self.number_verifier.verify_report(report, metrics)
            verified_reports.append(verified)

        # Count verified
        verified_count = sum(1 for r in verified_reports if r.verified)
        logger.info(
            "analyst_reports_verified",
            total=len(verified_reports),
            verified=verified_count
        )

        # Step 4: Judge synthesizes verified claims
        logger.info("judge_synthesizing")
        final_summary = await self._run_judge(date, verified_reports, metrics)

        # Step 5: Build council summary
        council_summary = CouncilSummary(
            date=date,
            final_summary=final_summary['summary'],
            key_learnings=final_summary['key_learnings'],
            recommendations=final_summary['recommendations'],
            hamilton_ready=final_summary['hamilton_ready'],
            analyst_reports=verified_reports,
            verification_status=f"{verified_count}/{len(verified_reports)} analysts verified",
            total_analysts=len(verified_reports),
            verified_analysts=verified_count
        )

        # Step 6: Cache result
        if self.enable_cache:
            self._save_to_cache(date, council_summary)

        logger.info(
            "daily_summary_complete",
            date=date,
            verified=verified_count,
            total=len(verified_reports)
        )

        return council_summary

    async def _run_analyst(
        self,
        analyst: Any,
        date: str,
        metrics: Dict[str, Any]
    ) -> AnalystReport:
        """Run individual analyst"""
        try:
            report = await analyst.analyze(date, metrics)
            return report
        except Exception as e:
            logger.error(
                "analyst_failed",
                analyst=analyst.name,
                error=str(e)
            )
            # Return empty report on failure
            return AnalystReport(
                analyst_name=analyst.name,
                model_name=analyst.model_name,
                summary=f"Analysis failed: {str(e)}",
                key_insights=[],
                numbers_cited={},
                verified=False,
                verification_errors=[f"Analyst failed: {str(e)}"]
            )

    async def _run_judge(
        self,
        date: str,
        analyst_reports: List[AnalystReport],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run judge to synthesize reports"""
        try:
            synthesis = await self.judge.synthesize(
                date=date,
                analyst_reports=analyst_reports,
                source_metrics=metrics
            )
            return synthesis
        except Exception as e:
            logger.error("judge_failed", error=str(e))
            # Fallback to simple aggregation
            return {
                'summary': f"Judge synthesis failed: {str(e)}",
                'key_learnings': [],
                'recommendations': [],
                'hamilton_ready': False
            }

    def _load_from_cache(self, date: str) -> Optional[CouncilSummary]:
        """Load summary from cache"""
        cache_file = self.cache_dir / f"summary_{date}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Reconstruct dataclasses
                analyst_reports = [
                    AnalystReport(**r) for r in data['analyst_reports']
                ]

                return CouncilSummary(
                    date=data['date'],
                    final_summary=data['final_summary'],
                    key_learnings=data['key_learnings'],
                    recommendations=data['recommendations'],
                    hamilton_ready=data['hamilton_ready'],
                    analyst_reports=analyst_reports,
                    verification_status=data['verification_status'],
                    total_analysts=data['total_analysts'],
                    verified_analysts=data['verified_analysts']
                )
            except Exception as e:
                logger.warning("cache_load_failed", date=date, error=str(e))
                return None
        return None

    def _save_to_cache(self, date: str, summary: CouncilSummary):
        """Save summary to cache"""
        cache_file = self.cache_dir / f"summary_{date}.json"
        try:
            # Convert to dict
            data = {
                'date': summary.date,
                'final_summary': summary.final_summary,
                'key_learnings': summary.key_learnings,
                'recommendations': summary.recommendations,
                'hamilton_ready': summary.hamilton_ready,
                'analyst_reports': [
                    {
                        'analyst_name': r.analyst_name,
                        'model_name': r.model_name,
                        'summary': r.summary,
                        'key_insights': r.key_insights,
                        'numbers_cited': r.numbers_cited,
                        'verified': r.verified,
                        'verification_errors': r.verification_errors
                    }
                    for r in summary.analyst_reports
                ],
                'verification_status': summary.verification_status,
                'total_analysts': summary.total_analysts,
                'verified_analysts': summary.verified_analysts
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("summary_cached", date=date, file=str(cache_file))
        except Exception as e:
            logger.warning("cache_save_failed", date=date, error=str(e))


if __name__ == '__main__':
    # Example usage (requires API keys)
    import os

    print("AI Council Manager Example")
    print("=" * 80)

    # Note: In production, load from environment or secure config
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY', 'demo-key'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY', 'demo-key'),
        'google': os.getenv('GOOGLE_API_KEY', 'demo-key'),
        'xai': os.getenv('XAI_API_KEY', 'demo-key'),
        'groq': os.getenv('GROQ_API_KEY', 'demo-key'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY', 'demo-key'),
        'perplexity': os.getenv('PERPLEXITY_API_KEY', 'demo-key')
    }

    print("\nNote: This is a demo. Real API keys required for actual use.")
    print("Set environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY, etc.")
    print("\nAI Council would:")
    print("  1. Pre-compute metrics (anti-hallucination)")
    print("  2. Run 8 analysts in parallel")
    print("  3. Verify all numbers cited")
    print("  4. Judge synthesizes verified claims")
    print("  5. Cache result for fast access")
    print("\n✓ AI Council Manager ready!")
