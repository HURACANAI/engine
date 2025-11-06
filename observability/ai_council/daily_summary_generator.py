"""
Daily Summary Generator

Convenience script for generating daily AI summaries.

Usage:
    # As a script
    python -m observability.ai_council.daily_summary_generator --date 2025-11-06

    # As a module
    from observability.ai_council.daily_summary_generator import generate_summary

    summary = await generate_summary(
        date='2025-11-06',
        api_keys={...}
    )
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Optional
import structlog

from observability.ai_council.council_manager import CouncilManager, CouncilSummary

logger = structlog.get_logger(__name__)


async def generate_summary(
    date: str,
    api_keys: Optional[Dict[str, str]] = None,
    force_refresh: bool = False
) -> CouncilSummary:
    """
    Generate daily summary using AI Council.

    Args:
        date: Date string (YYYY-MM-DD)
        api_keys: API keys for LLM providers (or load from env)
        force_refresh: Bypass cache

    Returns:
        CouncilSummary
    """
    # Load API keys from environment if not provided
    if api_keys is None:
        api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'google': os.getenv('GOOGLE_API_KEY'),
            'xai': os.getenv('XAI_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY'),
            'deepseek': os.getenv('DEEPSEEK_API_KEY')
        }

        # Filter out None values
        api_keys = {k: v for k, v in api_keys.items() if v}

    if not api_keys:
        raise ValueError(
            "No API keys provided. Set environment variables: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, etc."
        )

    # Initialize council
    council = CouncilManager(api_keys=api_keys)

    # Generate summary
    logger.info("generating_daily_summary", date=date, force_refresh=force_refresh)
    summary = await council.generate_daily_summary(date, force_refresh=force_refresh)

    return summary


def format_summary(summary: CouncilSummary) -> str:
    """Format summary for display"""
    output = []
    output.append("=" * 80)
    output.append(f"AI COUNCIL DAILY SUMMARY - {summary.date}")
    output.append("=" * 80)
    output.append("")

    # Final summary
    output.append("📝 SUMMARY:")
    output.append(f"  {summary.final_summary}")
    output.append("")

    # Key learnings
    if summary.key_learnings:
        output.append("🎓 KEY LEARNINGS:")
        for learning in summary.key_learnings:
            output.append(f"  • {learning}")
        output.append("")

    # Recommendations
    if summary.recommendations:
        output.append("💡 RECOMMENDATIONS:")
        for rec in summary.recommendations:
            output.append(f"  • {rec}")
        output.append("")

    # Hamilton readiness
    ready_icon = "✅" if summary.hamilton_ready else "⏳"
    output.append(f"🎯 HAMILTON READY: {ready_icon} {summary.hamilton_ready}")
    output.append("")

    # Verification status
    output.append(f"🔍 VERIFICATION: {summary.verification_status}")
    output.append("")

    # Analyst details (optional - uncomment to show)
    # output.append("📊 ANALYST REPORTS:")
    # for report in summary.analyst_reports:
    #     verified_icon = "✓" if report.verified else "✗"
    #     output.append(f"  {verified_icon} {report.analyst_name} ({report.model_name})")
    #     if report.verification_errors:
    #         for error in report.verification_errors:
    #             output.append(f"      ⚠️ {error}")
    # output.append("")

    output.append("=" * 80)

    return "\n".join(output)


async def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI Council daily summary")
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help='Date to analyze (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh (bypass cache)'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Save summary to file (e.g., summary.txt)'
    )

    args = parser.parse_args()

    try:
        # Generate summary
        summary = await generate_summary(date=args.date, force_refresh=args.force)

        # Format and display
        formatted = format_summary(summary)
        print(formatted)

        # Save if requested
        if args.save:
            with open(args.save, 'w') as f:
                f.write(formatted)
            print(f"\n✓ Summary saved to {args.save}")

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease set API keys as environment variables:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-...'")
        print("  export GOOGLE_API_KEY='AIza...'")
        print("  export XAI_API_KEY='...'")
        print("  export GROQ_API_KEY='...'")
        print("  export DEEPSEEK_API_KEY='...'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("daily_summary_failed")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
