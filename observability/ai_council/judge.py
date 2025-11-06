"""
Judge Model

Synthesizes verified analyst reports into final summary.

The judge:
1. Receives only VERIFIED analyst reports
2. Synthesizes common insights
3. Resolves disagreements
4. Produces final summary with recommendations

Uses Claude Opus for highest quality synthesis.
"""

import httpx
from typing import Dict, List, Any
import structlog
import json

from observability.ai_council.analysts.base_analyst import AnalystReport

logger = structlog.get_logger(__name__)


class JudgeModel:
    """
    Judge model for synthesizing analyst reports.

    Uses Claude Opus for highest quality reasoning and synthesis.
    """

    def __init__(self, api_key: str):
        """
        Initialize judge.

        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key
        self.model_name = "claude-3-opus-20240229"

        logger.info("judge_initialized", model=self.model_name)

    async def synthesize(
        self,
        date: str,
        analyst_reports: List[AnalystReport],
        source_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize analyst reports into final summary.

        Args:
            date: Date string (YYYY-MM-DD)
            analyst_reports: List of verified analyst reports
            source_metrics: Ground truth metrics (for final verification)

        Returns:
            Dict with:
                - summary: Final synthesized summary
                - key_learnings: List of key learnings
                - recommendations: List of recommendations
                - hamilton_ready: Boolean
        """
        logger.info(
            "judge_synthesizing",
            date=date,
            total_reports=len(analyst_reports),
            verified_reports=sum(1 for r in analyst_reports if r.verified)
        )

        # Filter to only verified reports
        verified_reports = [r for r in analyst_reports if r.verified]

        if not verified_reports:
            logger.warning("no_verified_reports")
            return {
                'summary': "No verified analyst reports available for synthesis.",
                'key_learnings': [],
                'recommendations': ["Review analyst report verification process"],
                'hamilton_ready': False
            }

        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(date, verified_reports, source_metrics)

        # Call Claude Opus
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            )

        result = response.json()
        response_text = result['content'][0]['text']

        # Parse synthesis
        synthesis = self._parse_synthesis(response_text)

        logger.info(
            "judge_synthesis_complete",
            learnings=len(synthesis['key_learnings']),
            recommendations=len(synthesis['recommendations'])
        )

        return synthesis

    def _build_synthesis_prompt(
        self,
        date: str,
        verified_reports: List[AnalystReport],
        source_metrics: Dict[str, Any]
    ) -> str:
        """Build synthesis prompt for judge"""

        # Format analyst reports
        reports_text = ""
        for i, report in enumerate(verified_reports, 1):
            reports_text += f"\n--- ANALYST {i}: {report.analyst_name} ({report.model_name}) ---\n"
            reports_text += f"SUMMARY: {report.summary}\n"
            reports_text += f"KEY INSIGHTS:\n"
            for insight in report.key_insights:
                reports_text += f"  - {insight}\n"
            reports_text += f"NUMBERS CITED (verified): {report.numbers_cited}\n"

        return f"""You are a senior AI judge synthesizing multiple analyst reports.

Today's date: {date}

Your task: Synthesize the VERIFIED analyst reports below into a final summary.

CRITICAL RULES:
1. Only use insights from VERIFIED reports (all numbers already verified)
2. Synthesize common themes across analysts
3. Resolve any disagreements by referencing source metrics
4. Be concise (3-5 sentences for summary)
5. NEVER introduce new numbers not cited by analysts

VERIFIED ANALYST REPORTS:
{reports_text}

SOURCE METRICS (for resolving disagreements):
{json.dumps(source_metrics, indent=2)}

Your synthesis should answer:
- What did the Engine learn today? (common theme)
- How did shadow trading perform? (paper only, no real money)
- Are models improving? (consensus view)
- Is the model ready for Hamilton? (final decision)
- What needs attention? (actionable recommendations)

Format your response as:
SUMMARY: (3-5 sentences synthesizing all analysts)
KEY_LEARNINGS: (bullet points - common themes)
RECOMMENDATIONS: (bullet points - actionable next steps)
HAMILTON_READY: (true or false)
"""

    def _parse_synthesis(self, response_text: str) -> Dict[str, Any]:
        """Parse judge's synthesis response"""
        lines = response_text.strip().split('\n')

        summary = ""
        key_learnings = []
        recommendations = []
        hamilton_ready = False

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("KEY_LEARNINGS:"):
                current_section = "learnings"
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            elif line.startswith("HAMILTON_READY:"):
                ready_text = line.replace("HAMILTON_READY:", "").strip().lower()
                hamilton_ready = ready_text in ['true', 'yes', '✓', 'ready']
            elif current_section == "summary":
                summary += " " + line
            elif current_section == "learnings" and line.startswith("-"):
                key_learnings.append(line.lstrip("- ").strip())
            elif current_section == "recommendations" and line.startswith("-"):
                recommendations.append(line.lstrip("- ").strip())

        return {
            'summary': summary.strip(),
            'key_learnings': key_learnings,
            'recommendations': recommendations,
            'hamilton_ready': hamilton_ready
        }


if __name__ == '__main__':
    # Example usage
    import os
    from dataclasses import dataclass

    print("Judge Model Example")
    print("=" * 80)

    @dataclass
    class MockAnalystReport:
        analyst_name: str
        model_name: str
        summary: str
        key_insights: List[str]
        numbers_cited: Dict[str, float]
        verified: bool
        verification_errors: List[str]

    # Mock reports
    reports = [
        MockAnalystReport(
            analyst_name="GPT-4",
            model_name="gpt-4-turbo",
            summary="Engine executed 42 shadow trades with 74% win rate. Trained 3 times with AUC improving to 0.72.",
            key_insights=[
                "Shadow trading active and performing well",
                "Model improving steadily",
                "Ready for Hamilton export soon"
            ],
            numbers_cited={"total_trades": 42, "win_rate": 0.74, "auc": 0.72},
            verified=True,
            verification_errors=[]
        ),
        MockAnalystReport(
            analyst_name="Claude",
            model_name="claude-sonnet",
            summary="42 shadow trades, 74% success rate. AUC at 0.72 after 3 training sessions.",
            key_insights=[
                "Strong shadow performance",
                "Model metrics trending positive",
                "Hamilton readiness criteria nearly met"
            ],
            numbers_cited={"total_trades": 42, "win_rate": 0.74, "auc": 0.72},
            verified=True,
            verification_errors=[]
        )
    ]

    metrics = {
        "shadow_trading": {"total_trades": 42, "win_rate": 0.74},
        "learning": {"num_sessions": 3, "best_auc": 0.72}
    }

    print("\nNote: This is a demo. Real API key required for actual use.")
    print(f"Set environment variable: ANTHROPIC_API_KEY")
    print("\nJudge would:")
    print(f"  1. Receive {len(reports)} verified analyst reports")
    print(f"  2. Synthesize common themes (e.g., '42 shadow trades, 74% win rate')")
    print(f"  3. Resolve disagreements using source metrics")
    print(f"  4. Produce final summary with recommendations")
    print(f"  5. Decide Hamilton readiness")
    print("\n✓ Judge Model ready!")
