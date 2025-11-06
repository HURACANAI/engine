"""
DeepSeek Analyst

Uses DeepSeek-R1 for analysis.
"""

import httpx
from typing import Dict, Any
import structlog

from observability.ai_council.analysts.base_analyst import BaseAnalyst, AnalystReport

logger = structlog.get_logger(__name__)


class DeepSeekAnalyst(BaseAnalyst):
    """Analyst using DeepSeek-R1"""

    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            name="DeepSeek Analyst",
            model_name="deepseek-reasoner"
        )

    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """Analyze metrics using DeepSeek"""
        logger.info("deepseek_analyzing", date=date)

        prompt = self._build_prompt(date, metrics)

        # Call DeepSeek API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-reasoner",
                    "messages": [
                        {"role": "system", "content": "You are an expert AI analyst for machine learning systems. Be precise and never invent numbers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 500
                }
            )

        result = response.json()
        response_text = result['choices'][0]['message']['content']

        # Parse response
        report = self._parse_response(response_text)

        logger.info(
            "deepseek_analysis_complete",
            numbers_cited=len(report.numbers_cited),
            insights=len(report.key_insights)
        )

        return report
