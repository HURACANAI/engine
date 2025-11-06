"""
Perplexity Analyst

Uses Perplexity's API for analysis.
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class PerplexityAnalyst(BaseAnalyst):
    """Analyst using Perplexity AI"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="Perplexity Analyst",
            model_name="llama-3.1-sonar-large-128k-online"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using Perplexity"""
        logger.info("perplexity_analyzing", date=date)

        prompt = self._build_prompt(date, metrics)

        # Call Perplexity API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [
                        {"role": "system", "content": "You are an expert AI analyst for machine learning systems. Be precise and never invent numbers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,  # Deterministic
                    "max_tokens": 500
                }
            )

        result = response.json()
        response_text = result['choices'][0]['message']['content']

        # Parse response
        report = self._parse_response(response_text)

        logger.info(
            "perplexity_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
