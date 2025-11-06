"""
Base Analyst

Abstract base class for all AI analyst models.

All analysts follow the same interface but use different LLMs for diverse reasoning.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AnalystReport:
    """Individual analyst's report"""
    analyst_name: str
    model_name: str
    summary: str
    key_insights: List[str]
    numbers_cited: Dict[str, float]
    verified: bool = False
    verification_errors: List[str] = None

    def __post_init__(self):
        if self.verification_errors is None:
            self.verification_errors = []


class BaseAnalyst(ABC):
    """
    Base class for all analyst models.

    Each analyst:
    1. Receives pre-computed metrics (no raw logs)
    2. Analyzes independently
    3. Declares all numbers cited (for verification)
    4. Returns structured report
    """

    def __init__(self, api_key: str, name: str, model_name: str):
        """
        Initialize analyst.

        Args:
            api_key: API key for the LLM provider
            name: Human-readable analyst name
            model_name: Model identifier
        """
        self.api_key = api_key
        self.name = name
        self.model_name = model_name

        logger.info("analyst_initialized", name=name, model=model_name)

    @abstractmethod
    async def analyze(self, date: str, metrics: Dict[str, Any]) -> AnalystReport:
        """
        Analyze metrics and produce report.

        Args:
            date: Date string (YYYY-MM-DD)
            metrics: Pre-computed metrics from MetricsComputer

        Returns:
            AnalystReport with summary and cited numbers
        """
        pass

    def _build_prompt(self, date: str, metrics: Dict[str, Any]) -> str:
        """
        Build analysis prompt.

        All analysts use the same prompt structure (anti-hallucination).
        """
        return f"""You are an expert AI analyst for a machine learning trading system.

Today's date: {date}

Your task: Analyze the Engine's learning progress and provide insights.

CRITICAL RULES:
1. NEVER invent numbers - only use numbers from the metrics below
2. Declare ALL numbers you cite in your response
3. Focus on: learning progress, shadow trade performance, model improvements
4. Be concise (3-5 sentences max)

METRICS (Pre-computed - these are the ONLY valid numbers):
{self._format_metrics(metrics)}

Your analysis should answer:
- What did the Engine learn today?
- How did shadow trading perform? (paper only, no real money)
- Are models improving?
- Is the model ready for Hamilton export?
- What needs attention?

Format your response as:
SUMMARY: (3-5 sentences)
KEY_INSIGHTS: (bullet points)
NUMBERS_CITED: (key: value pairs for every number you mentioned)
"""

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for prompt"""
        import json
        return json.dumps(metrics, indent=2)

    def _parse_response(self, response_text: str) -> AnalystReport:
        """
        Parse LLM response into AnalystReport.

        Expected format:
        SUMMARY: ...
        KEY_INSIGHTS:
        - ...
        NUMBERS_CITED:
        key: value
        """
        lines = response_text.strip().split('\n')

        summary = ""
        key_insights = []
        numbers_cited = {}

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("KEY_INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("NUMBERS_CITED:"):
                current_section = "numbers"
            elif current_section == "summary":
                summary += " " + line
            elif current_section == "insights" and line.startswith("-"):
                key_insights.append(line.lstrip("- ").strip())
            elif current_section == "numbers" and ":" in line:
                try:
                    key, value = line.split(":", 1)
                    numbers_cited[key.strip()] = float(value.strip())
                except ValueError:
                    logger.warning("failed_to_parse_number", line=line)

        return AnalystReport(
            analyst_name=self.name,
            model_name=self.model_name,
            summary=summary.strip(),
            key_insights=key_insights,
            numbers_cited=numbers_cited
        )
