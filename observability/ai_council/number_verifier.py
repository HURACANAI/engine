"""
Number Verifier

Prevents AI hallucination by verifying every number cited in analyst reports.

This is the critical anti-hallucination layer:
- Extract all numbers from analyst summary
- Compare against source metrics
- Flag any invented numbers
- Block reports with hallucinations

Usage:
    verifier = NumberVerifier()

    verified_report = verifier.verify_report(analyst_report, source_metrics)

    if verified_report.verified:
        print("✓ All numbers verified")
    else:
        print(f"✗ Verification errors: {verified_report.verification_errors}")
"""

import re
from typing import Dict, List, Any, Set, Tuple
import structlog

logger = structlog.get_logger(__name__)


class NumberVerifier:
    """
    Verify all numbers cited in AI reports.

    Prevents hallucination by checking every number against source metrics.
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize number verifier.

        Args:
            tolerance: Acceptable difference (0.01 = 1%)
        """
        self.tolerance = tolerance
        logger.info("number_verifier_initialized", tolerance=tolerance)

    def verify_report(self, report: Any, source_metrics: Dict[str, Any]) -> Any:
        """
        Verify analyst report against source metrics.

        Args:
            report: AnalystReport with numbers_cited
            source_metrics: Ground truth metrics from MetricsComputer

        Returns:
            Modified AnalystReport with verification results
        """
        logger.debug(
            "verifying_report",
            analyst=report.analyst_name,
            numbers_count=len(report.numbers_cited)
        )

        errors = []

        # Flatten source metrics for easy lookup
        flat_metrics = self._flatten_metrics(source_metrics)

        # Check each number cited
        for key, cited_value in report.numbers_cited.items():
            # Find matching metric
            source_value = self._find_metric(key, flat_metrics)

            if source_value is None:
                errors.append(f"Number '{key}' not found in source metrics (possible hallucination)")
            else:
                # Compare with tolerance
                if not self._values_match(cited_value, source_value, self.tolerance):
                    errors.append(
                        f"Number mismatch: {key} = {cited_value} (cited) vs {source_value} (source)"
                    )

        # Also check for numbers in summary text that weren't declared
        undeclared = self._find_undeclared_numbers(report.summary, report.numbers_cited)
        if undeclared:
            errors.append(f"Undeclared numbers in summary: {undeclared}")

        # Update report
        report.verified = len(errors) == 0
        report.verification_errors = errors

        if report.verified:
            logger.info("report_verified", analyst=report.analyst_name)
        else:
            logger.warning(
                "report_verification_failed",
                analyst=report.analyst_name,
                errors=len(errors)
            )

        return report

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Flatten nested metrics dict for easy lookup"""
        flat = {}

        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse
                flat.update(self._flatten_metrics(value, full_key))
            elif isinstance(value, (int, float)):
                flat[full_key] = float(value)
            elif isinstance(value, list):
                # Handle lists (e.g., gate metrics)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat.update(self._flatten_metrics(item, f"{full_key}[{i}]"))

        return flat

    def _find_metric(self, key: str, flat_metrics: Dict[str, float]) -> float | None:
        """Find metric by key (fuzzy matching)"""
        # Exact match
        if key in flat_metrics:
            return flat_metrics[key]

        # Try case-insensitive
        key_lower = key.lower()
        for metric_key, value in flat_metrics.items():
            if metric_key.lower() == key_lower:
                return value

        # Try partial match (e.g., "win_rate" matches "shadow_trading.win_rate")
        for metric_key, value in flat_metrics.items():
            if key_lower in metric_key.lower() or metric_key.lower() in key_lower:
                return value

        return None

    def _values_match(self, cited: float, source: float, tolerance: float) -> bool:
        """Check if two values match within tolerance"""
        if source == 0:
            # For zero values, check absolute difference
            return abs(cited - source) <= tolerance
        else:
            # Check percentage difference
            pct_diff = abs((cited - source) / source)
            return pct_diff <= tolerance

    def _find_undeclared_numbers(
        self,
        text: str,
        declared_numbers: Dict[str, float]
    ) -> List[str]:
        """Find numbers in text that weren't declared in numbers_cited"""
        # Extract all numbers from text
        # Patterns: 42, 42.5, 42%, 0.42, etc.
        number_pattern = r'\b\d+\.?\d*%?\b'
        matches = re.findall(number_pattern, text)

        # Convert to floats
        text_numbers = set()
        for match in matches:
            try:
                # Remove % if present
                value_str = match.rstrip('%')
                value = float(value_str)
                # If original had %, it was a percentage
                if match.endswith('%'):
                    value = value / 100.0
                text_numbers.add(value)
            except ValueError:
                pass

        # Check which text numbers aren't in declared
        declared_values = set(declared_numbers.values())
        undeclared = []

        for text_num in text_numbers:
            # Check if this number is close to any declared number
            found = False
            for declared_val in declared_values:
                if self._values_match(text_num, declared_val, self.tolerance):
                    found = True
                    break

            if not found:
                undeclared.append(str(text_num))

        return undeclared


if __name__ == '__main__':
    # Example usage
    print("Number Verifier Example")
    print("=" * 80)

    from dataclasses import dataclass

    @dataclass
    class MockAnalystReport:
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

    verifier = NumberVerifier(tolerance=0.01)

    # Source metrics (ground truth)
    source_metrics = {
        "shadow_trading": {
            "total_trades": 42,
            "win_rate": 0.74,
            "avg_pnl_bps": 5.3
        },
        "learning": {
            "num_sessions": 3,
            "best_auc": 0.72
        }
    }

    print("\n📊 Source Metrics:")
    print(f"  Shadow trades: {source_metrics['shadow_trading']['total_trades']}")
    print(f"  Win rate: {source_metrics['shadow_trading']['win_rate']:.1%}")
    print(f"  Training sessions: {source_metrics['learning']['num_sessions']}")

    # Test 1: Correct report
    print("\n✅ Test 1: Correct Report")
    report1 = MockAnalystReport(
        analyst_name="GPT-4",
        model_name="gpt-4-turbo",
        summary="Engine executed 42 shadow trades with 74% win rate. Trained 3 times.",
        key_insights=["Good performance"],
        numbers_cited={
            "total_trades": 42,
            "win_rate": 0.74,
            "num_sessions": 3
        }
    )

    verified1 = verifier.verify_report(report1, source_metrics)
    print(f"  Verified: {verified1.verified}")
    if not verified1.verified:
        for error in verified1.verification_errors:
            print(f"    ✗ {error}")

    # Test 2: Hallucinated number
    print("\n❌ Test 2: Hallucinated Number")
    report2 = MockAnalystReport(
        analyst_name="Claude",
        model_name="claude-sonnet",
        summary="Engine executed 100 shadow trades with 90% win rate.",
        key_insights=["Excellent performance"],
        numbers_cited={
            "total_trades": 100,  # WRONG (actual = 42)
            "win_rate": 0.90  # WRONG (actual = 0.74)
        }
    )

    verified2 = verifier.verify_report(report2, source_metrics)
    print(f"  Verified: {verified2.verified}")
    if not verified2.verified:
        for error in verified2.verification_errors:
            print(f"    ✗ {error}")

    # Test 3: Undeclared numbers
    print("\n⚠️  Test 3: Undeclared Numbers")
    report3 = MockAnalystReport(
        analyst_name="Gemini",
        model_name="gemini-1.5-pro",
        summary="Engine improved by 42% over last week.",  # 42% not declared
        key_insights=["Big improvement"],
        numbers_cited={}  # Empty!
    )

    verified3 = verifier.verify_report(report3, source_metrics)
    print(f"  Verified: {verified3.verified}")
    if not verified3.verified:
        for error in verified3.verification_errors:
            print(f"    ✗ {error}")

    print("\n✓ Number Verifier ready!")
    print("This prevents AI hallucination by checking every number.")
