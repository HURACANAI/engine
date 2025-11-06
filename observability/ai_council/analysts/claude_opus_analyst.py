"""
Claude Opus Analyst

Uses Anthropic's Claude 3 Opus for analysis (as an analyst, not judge).
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class ClaudeOpusAnalyst(BaseAnalyst):
    """Analyst using Claude 3 Opus"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="Claude Opus Analyst",
            model_name="claude-3-opus-20240229"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using Claude Opus"""
        logger.info("claude_opus_analyzing", date=date)

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
                    "model": "claude-3-opus-20240229",
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
            "claude_opus_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
