"""
GPT-4 Analyst

Uses OpenAI's GPT-4-Turbo for analysis.
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class GPT4Analyst(BaseAnalyst):
    """Analyst using GPT-4-Turbo"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="GPT-4 Analyst",
            model_name="gpt-4-turbo-preview"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using GPT-4"""
        logger.info("gpt4_analyzing", date=date)

        prompt = self._build_prompt(date, metrics)

        # Call OpenAI API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4-turbo-preview",
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
            "gpt4_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
