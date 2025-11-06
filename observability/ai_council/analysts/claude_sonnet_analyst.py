"""
Claude Sonnet Analyst

Uses Anthropic's Claude 3.5 Sonnet for analysis.
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class ClaudeSonnetAnalyst(BaseAnalyst):
    """Analyst using Claude 3.5 Sonnet"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="Claude Sonnet Analyst",
            model_name="claude-3-5-sonnet-20241022"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using Claude Sonnet"""
        logger.info("claude_sonnet_analyzing", date=date)

        prompt = self._build_prompt(date, metrics)

        # Call Anthropic API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500
                }
            )

        result = response.json()
        response_text = result['content'][0]['text']

        # Parse response
        report = self._parse_response(response_text)

        logger.info(
            "claude_sonnet_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
