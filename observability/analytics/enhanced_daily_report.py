"""
Enhanced Daily Learning Report Generator

Comprehensive daily report with actionable insights:
1. WHAT IT LEARNED TODAY:
   - New patterns discovered
   - Features that became more/less important
   - Regime-specific improvements
   - Model improvements (AUC delta, calibration)

2. WHAT CHANGED:
   - Model updates (before/after metrics)
   - Gate threshold adjustments
   - New strategies enabled/disabled
   - Configuration changes

3. PERFORMANCE SUMMARY:
   - Shadow trades executed
   - Win rate by mode/regime
   - P&L breakdown
   - Best/worst performing strategies

4. ISSUES & ALERTS:
   - Anomalies detected
   - Degrading patterns
   - Model drift warnings
   - Recommendations for fixes

5. NEXT STEPS:
   - What to monitor tomorrow
   - Suggested parameter tweaks
   - Patterns to investigate

Usage:
    python -m observability.analytics.enhanced_daily_report --date 2025-01-XX
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog
from dataclasses import dataclass

from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.model_evolution import ModelEvolutionTracker
from observability.analytics.insight_aggregator import InsightAggregator
from observability.analytics.gate_explainer import GateExplainer

logger = structlog.get_logger(__name__)


@dataclass
class DailyReport:
    """Complete daily learning report"""
    date: str
    summary: str
    
    # What it learned
    new_patterns: List[str]
    feature_changes: List[Dict[str, Any]]
    regime_improvements: Dict[str, Dict[str, float]]
    model_improvements: Dict[str, float]
    
    # What changed
    model_updates: List[Dict[str, Any]]
    gate_changes: List[Dict[str, Any]]
    strategy_changes: List[str]
    config_changes: List[str]
    
    # Performance
    shadow_trades: Dict[str, Any]
    win_rate_by_mode: Dict[str, float]
    win_rate_by_regime: Dict[str, float]
    pnl_breakdown: Dict[str, Any]
    best_strategies: List[Dict[str, Any]]
    worst_strategies: List[Dict[str, Any]]
    
    # Issues & Alerts
    anomalies: List[Dict[str, Any]]
    degrading_patterns: List[Dict[str, Any]]
    drift_warnings: List[Dict[str, Any]]
    recommendations: List[str]
    
    # Next steps
    monitor_tomorrow: List[str]
    suggested_tweaks: List[Dict[str, Any]]
    patterns_to_investigate: List[str]


class EnhancedDailyReportGenerator:
    """
    Generate comprehensive daily learning reports.
    
    Combines all observability systems to provide complete picture.
    """

    def __init__(self):
        """Initialize report generator"""
        self.learning_tracker = LearningTracker()
        self.trade_journal = TradeJournal()
        self.metrics_computer = MetricsComputer()
        self.model_tracker = ModelEvolutionTracker()
        self.insight_aggregator = InsightAggregator()
        self.gate_explainer = GateExplainer()

        logger.info("enhanced_daily_report_generator_initialized")

    def generate_report(self, date: str) -> DailyReport:
        """
        Generate complete daily report.
        
        Args:
            date: Date string (YYYY-MM-DD)
            
        Returns:
            DailyReport with all insights
        """
        logger.info("generating_enhanced_daily_report", date=date)

        # 1. What it learned today
        learning_summary = self.learning_tracker.get_daily_summary(date)
        new_patterns = self._extract_new_patterns(date)
        feature_changes = self._extract_feature_changes(date)
        regime_improvements = self._extract_regime_improvements(date)
        model_improvements = self._extract_model_improvements(date)

        # 2. What changed
        model_updates = self._extract_model_updates(date)
        gate_changes = self._extract_gate_changes(date)
        strategy_changes = self._extract_strategy_changes(date)
        config_changes = self._extract_config_changes(date)

        # 3. Performance summary
        shadow_trades = self._get_shadow_trade_summary(date)
        win_rate_by_mode = self._get_win_rate_by_mode(date)
        win_rate_by_regime = self._get_win_rate_by_regime(date)
        pnl_breakdown = self._get_pnl_breakdown(date)
        best_strategies, worst_strategies = self._get_strategy_performance(date)

        # 4. Issues & Alerts
        anomalies = self._detect_anomalies(date)
        degrading_patterns = self._detect_degrading_patterns(date)
        drift_warnings = self._detect_drift_warnings(date)
        recommendations = self._generate_recommendations(
            learning_summary, shadow_trades, model_improvements, anomalies, degrading_patterns
        )

        # 5. Next steps
        monitor_tomorrow = self._suggest_monitoring(date)
        suggested_tweaks = self._suggest_tweaks(date)
        patterns_to_investigate = self._suggest_investigations(date)

        # Generate summary
        summary = self._generate_summary(
            shadow_trades, model_improvements, recommendations
        )

        return DailyReport(
            date=date,
            summary=summary,
            new_patterns=new_patterns,
            feature_changes=feature_changes,
            regime_improvements=regime_improvements,
            model_improvements=model_improvements,
            model_updates=model_updates,
            gate_changes=gate_changes,
            strategy_changes=strategy_changes,
            config_changes=config_changes,
            shadow_trades=shadow_trades,
            win_rate_by_mode=win_rate_by_mode,
            win_rate_by_regime=win_rate_by_regime,
            pnl_breakdown=pnl_breakdown,
            best_strategies=best_strategies,
            worst_strategies=worst_strategies,
            anomalies=anomalies,
            degrading_patterns=degrading_patterns,
            drift_warnings=drift_warnings,
            recommendations=recommendations,
            monitor_tomorrow=monitor_tomorrow,
            suggested_tweaks=suggested_tweaks,
            patterns_to_investigate=patterns_to_investigate,
        )

    def format_report(self, report: DailyReport) -> str:
        """Format report as markdown"""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"ENGINE DAILY LEARNING REPORT - {report.date}")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("## 📋 EXECUTIVE SUMMARY")
        lines.append("")
        lines.append(report.summary)
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # What it learned
        lines.append("## 🎓 WHAT IT LEARNED TODAY")
        lines.append("")
        
        if report.new_patterns:
            lines.append("### New Patterns Discovered")
            for pattern in report.new_patterns:
                lines.append(f"- {pattern}")
            lines.append("")
        
        if report.feature_changes:
            lines.append("### Feature Importance Changes")
            for change in report.feature_changes[:10]:  # Top 10
                feature = change['feature']
                delta = change['delta']
                direction = "↑" if delta > 0 else "↓"
                lines.append(f"- {feature}: {direction} {abs(delta):.1%}")
            lines.append("")
        
        if report.model_improvements:
            lines.append("### Model Improvements")
            for metric, value in report.model_improvements.items():
                sign = "+" if value > 0 else ""
                lines.append(f"- {metric}: {sign}{value:.3f}")
            lines.append("")
        
        if report.regime_improvements:
            lines.append("### Regime-Specific Improvements")
            for regime, metrics in report.regime_improvements.items():
                lines.append(f"- **{regime}**:")
                for metric, value in metrics.items():
                    sign = "+" if value > 0 else ""
                    lines.append(f"  - {metric}: {sign}{value:.3f}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # What changed
        lines.append("## 🔧 WHAT CHANGED")
        lines.append("")
        
        if report.model_updates:
            lines.append("### Model Updates")
            for update in report.model_updates:
                model_id = update.get('model_id', 'Unknown')
                before = update.get('before', {})
                after = update.get('after', {})
                lines.append(f"- **Model {model_id[:8]}...**")
                for metric in ['auc', 'ece', 'brier']:
                    if metric in before and metric in after:
                        delta = after[metric] - before[metric]
                        sign = "+" if delta > 0 else ""
                        lines.append(f"  - {metric.upper()}: {before[metric]:.3f} → {after[metric]:.3f} ({sign}{delta:.3f})")
            lines.append("")
        
        if report.gate_changes:
            lines.append("### Gate Threshold Adjustments")
            for change in report.gate_changes:
                gate = change.get('gate', 'Unknown')
                before = change.get('before', 0)
                after = change.get('after', 0)
                lines.append(f"- {gate}: {before:.3f} → {after:.3f}")
            lines.append("")
        
        if report.strategy_changes:
            lines.append("### Strategy Changes")
            for change in report.strategy_changes:
                lines.append(f"- {change}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Performance
        lines.append("## 📊 PERFORMANCE SUMMARY")
        lines.append("")
        
        shadow = report.shadow_trades
        lines.append(f"### Shadow Trading (Paper Only)")
        lines.append(f"- Total trades: {shadow.get('total_trades', 0)}")
        lines.append(f"- Win rate: {shadow.get('win_rate', 0):.1%}")
        lines.append(f"- Avg P&L: {shadow.get('avg_pnl_bps', 0):.1f} bps")
        lines.append(f"- Total P&L: £{shadow.get('total_pnl_gbp', 0):.2f} (simulated)")
        lines.append("")
        
        if report.win_rate_by_mode:
            lines.append("### Win Rate by Mode")
            for mode, wr in report.win_rate_by_mode.items():
                lines.append(f"- {mode}: {wr:.1%}")
            lines.append("")
        
        if report.win_rate_by_regime:
            lines.append("### Win Rate by Regime")
            for regime, wr in report.win_rate_by_regime.items():
                lines.append(f"- {regime}: {wr:.1%}")
            lines.append("")
        
        if report.best_strategies:
            lines.append("### Best Performing Strategies")
            for strategy in report.best_strategies[:5]:
                name = strategy.get('name', 'Unknown')
                wr = strategy.get('win_rate', 0)
                pnl = strategy.get('avg_pnl_bps', 0)
                lines.append(f"- {name}: {wr:.1%} WR, {pnl:.1f} bps avg")
            lines.append("")
        
        if report.worst_strategies:
            lines.append("### Worst Performing Strategies")
            for strategy in report.worst_strategies[:5]:
                name = strategy.get('name', 'Unknown')
                wr = strategy.get('win_rate', 0)
                pnl = strategy.get('avg_pnl_bps', 0)
                lines.append(f"- {name}: {wr:.1%} WR, {pnl:.1f} bps avg")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Issues & Alerts
        lines.append("## ⚠️ ISSUES & ALERTS")
        lines.append("")
        
        if report.anomalies:
            lines.append("### Anomalies Detected")
            for anomaly in report.anomalies:
                severity = anomaly.get('severity', 'WARNING')
                message = anomaly.get('message', 'Unknown')
                lines.append(f"- **{severity}**: {message}")
            lines.append("")
        
        if report.degrading_patterns:
            lines.append("### Degrading Patterns")
            for pattern in report.degrading_patterns:
                name = pattern.get('name', 'Unknown')
                wr = pattern.get('win_rate', 0)
                trend = pattern.get('trend', 'Unknown')
                lines.append(f"- {name}: {wr:.1%} WR ({trend})")
            lines.append("")
        
        if report.drift_warnings:
            lines.append("### Concept Drift Warnings")
            for warning in report.drift_warnings:
                component = warning.get('component', 'Unknown')
                severity = warning.get('severity', 'WARNING')
                lines.append(f"- **{component}**: {severity}")
            lines.append("")
        
        if report.recommendations:
            lines.append("### Recommendations")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # Next steps
        lines.append("## 🎯 NEXT STEPS")
        lines.append("")
        
        if report.monitor_tomorrow:
            lines.append("### Monitor Tomorrow")
            for item in report.monitor_tomorrow:
                lines.append(f"- {item}")
            lines.append("")
        
        if report.suggested_tweaks:
            lines.append("### Suggested Parameter Tweaks")
            for tweak in report.suggested_tweaks:
                param = tweak.get('parameter', 'Unknown')
                current = tweak.get('current', 0)
                suggested = tweak.get('suggested', 0)
                reason = tweak.get('reason', '')
                lines.append(f"- **{param}**: {current} → {suggested} ({reason})")
            lines.append("")
        
        if report.patterns_to_investigate:
            lines.append("### Patterns to Investigate")
            for pattern in report.patterns_to_investigate:
                lines.append(f"- {pattern}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

    # Helper methods (implementations would query actual data)
    def _extract_new_patterns(self, date: str) -> List[str]:
        """Extract newly discovered patterns"""
        # TODO: Implement pattern discovery tracking
        return []

    def _extract_feature_changes(self, date: str) -> List[Dict[str, Any]]:
        """Extract feature importance changes"""
        try:
            summary = self.learning_tracker.get_daily_summary(date)
            # TODO: Compare with previous day
            return []
        except Exception:
            return []

    def _extract_regime_improvements(self, date: str) -> Dict[str, Dict[str, float]]:
        """Extract regime-specific improvements"""
        # TODO: Compare regime performance
        return {}

    def _extract_model_improvements(self, date: str) -> Dict[str, float]:
        """Extract model metric improvements"""
        try:
            summary = self.learning_tracker.get_daily_summary(date)
            return {
                'auc_delta': summary.get('auc_delta', 0),
                'ece_delta': summary.get('ece_delta', 0),
            }
        except Exception:
            return {}

    def _extract_model_updates(self, date: str) -> List[Dict[str, Any]]:
        """Extract model version updates"""
        # TODO: Track model version changes
        return []

    def _extract_gate_changes(self, date: str) -> List[Dict[str, Any]]:
        """Extract gate threshold changes"""
        # TODO: Track gate threshold history
        return []

    def _extract_strategy_changes(self, date: str) -> List[str]:
        """Extract strategy enable/disable changes"""
        # TODO: Track strategy changes
        return []

    def _extract_config_changes(self, date: str) -> List[str]:
        """Extract configuration changes"""
        # TODO: Track config changes
        return []

    def _get_shadow_trade_summary(self, date: str) -> Dict[str, Any]:
        """Get shadow trade summary"""
        try:
            stats = self.trade_journal.get_stats(days=1)
            return {
                'total_trades': stats.get('total_trades', 0),
                'win_rate': stats.get('win_rate', 0),
                'avg_pnl_bps': stats.get('avg_return_bps', 0),
                'total_pnl_gbp': stats.get('total_pnl_gbp', 0),
            }
        except Exception:
            return {}

    def _get_win_rate_by_mode(self, date: str) -> Dict[str, float]:
        """Get win rate by trading mode"""
        # TODO: Query by mode
        return {}

    def _get_win_rate_by_regime(self, date: str) -> Dict[str, float]:
        """Get win rate by market regime"""
        # TODO: Query by regime
        return {}

    def _get_pnl_breakdown(self, date: str) -> Dict[str, Any]:
        """Get P&L breakdown"""
        # TODO: Calculate breakdown
        return {}

    def _get_strategy_performance(self, date: str) -> tuple[List[Dict], List[Dict]]:
        """Get best and worst performing strategies"""
        # TODO: Query strategy performance
        return [], []

    def _detect_anomalies(self, date: str) -> List[Dict[str, Any]]:
        """Detect anomalies"""
        # TODO: Integrate with anomaly detector
        return []

    def _detect_degrading_patterns(self, date: str) -> List[Dict[str, Any]]:
        """Detect degrading patterns"""
        # TODO: Integrate with pattern health monitor
        return []

    def _detect_drift_warnings(self, date: str) -> List[Dict[str, Any]]:
        """Detect concept drift warnings"""
        # TODO: Integrate with drift detector
        return []

    def _generate_recommendations(
        self,
        learning: Dict,
        shadow_trades: Dict,
        model_improvements: Dict,
        anomalies: List,
        degrading_patterns: List,
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check shadow trading activity
        total_trades = shadow_trades.get('total_trades', 0)
        if total_trades == 0:
            recommendations.append(
                "⚠️ NO SHADOW TRADES - Gates are too strict. Consider lowering meta_label threshold."
            )
        elif total_trades < 20:
            recommendations.append(
                "📊 Low shadow trading volume - Consider loosening gate thresholds to increase learning data."
            )
        
        # Check model improvements
        auc_delta = model_improvements.get('auc_delta', 0)
        if auc_delta > 0.02:
            recommendations.append(
                f"✅ Significant model improvement (+{auc_delta:.3f} AUC) - Consider exporting to Hamilton."
            )
        elif auc_delta < -0.01:
            recommendations.append(
                f"🚨 Model regression detected ({auc_delta:.3f} AUC) - Investigate immediately."
            )
        
        # Check anomalies
        if anomalies:
            recommendations.append(
                f"⚠️ {len(anomalies)} anomaly(ies) detected - Review and investigate."
            )
        
        # Check degrading patterns
        if degrading_patterns:
            recommendations.append(
                f"📉 {len(degrading_patterns)} pattern(s) degrading - Consider pausing or adjusting."
            )
        
        if not recommendations:
            recommendations.append("✓ All systems operating normally")
        
        return recommendations

    def _suggest_monitoring(self, date: str) -> List[str]:
        """Suggest what to monitor tomorrow"""
        return [
            "Monitor model performance for continued improvement",
            "Watch for concept drift in market conditions",
            "Track gate pass rates for optimal tuning",
        ]

    def _suggest_tweaks(self, date: str) -> List[Dict[str, Any]]:
        """Suggest parameter tweaks"""
        # TODO: Implement intelligent tweak suggestions
        return []

    def _suggest_investigations(self, date: str) -> List[str]:
        """Suggest patterns to investigate"""
        # TODO: Implement investigation suggestions
        return []

    def _generate_summary(
        self,
        shadow_trades: Dict,
        model_improvements: Dict,
        recommendations: List[str],
    ) -> str:
        """Generate executive summary"""
        total_trades = shadow_trades.get('total_trades', 0)
        win_rate = shadow_trades.get('win_rate', 0)
        auc_delta = model_improvements.get('auc_delta', 0)
        
        if total_trades == 0:
            return (
                f"⚠️ NO LEARNING TODAY - Engine received signals but gates blocked all trades. "
                f"Model AUC change: {auc_delta:+.3f}. Action needed: Loosen gate thresholds."
            )
        else:
            return (
                f"✅ ACTIVE LEARNING - {total_trades} shadow trades executed, "
                f"win rate {win_rate:.1%}. Model AUC change: {auc_delta:+.3f}. "
                f"{len(recommendations)} recommendation(s) provided."
            )


def main():
    """Main entry point"""
    import sys
    from datetime import datetime
    
    # Get date from args or use today
    if len(sys.argv) > 1 and sys.argv[1] == '--date':
        date = sys.argv[2]
    else:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    
    generator = EnhancedDailyReportGenerator()
    report = generator.generate_report(date)
    
    formatted = generator.format_report(report)
    print(formatted)
    
    # Also save to file
    output_path = f"observability/data/reports/daily_report_{date}.md"
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(formatted)
    
    print(f"\n✓ Report saved to: {output_path}")


if __name__ == '__main__':
    main()

