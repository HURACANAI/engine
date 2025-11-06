"""
Gemini Analyst

Uses Google's Gemini 1.5 Pro for analysis.
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class GeminiAnalyst(BaseAnalyst):
    """Analyst using Gemini 1.5 Pro"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="Gemini Analyst",
            model_name="gemini-1.5-pro"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using Gemini"""
        logger.info("gemini_analyzing", date=date)

        prompt = self._build_prompt(date, metrics)

        # Call Google Gemini API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.api_key}",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 500
                    }
                }
            )

        result = response.json()
        response_text = result['candidates'][0]['content']['parts'][0]['text']

        # Parse response
        report = self._parse_response(response_text)

        logger.info(
            "gemini_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
