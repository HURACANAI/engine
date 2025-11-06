"""
Insight Aggregator

Combines insights from all observability systems to answer:
- "What did the Engine learn today?"
- "Are models improving?"
- "Are models ready for Hamilton?"
- "What needs attention?"

Usage:
    aggregator = InsightAggregator()

    # Daily insights
    insights = aggregator.get_daily_insights(date="2025-11-06")

    # Shows:
    # - Shadow trade performance
    # - Model improvements
    # - Gate tuning recommendations
    # - Hamilton readiness
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.model_evolution import ModelEvolutionTracker

logger = structlog.get_logger(__name__)


class InsightAggregator:
    """
    Aggregate insights from all observability systems.

    Combines:
    - Learning progress (learning_tracker)
    - Shadow trade performance (trade_journal)
    - Model improvements (model_evolution)
    - Pre-computed metrics (metrics_computer)
    """

    def __init__(self):
        """Initialize insight aggregator"""
        self.learning_tracker = LearningTracker()
        self.trade_journal = TradeJournal()
        self.metrics_computer = MetricsComputer()
        self.model_evolution = ModelEvolutionTracker()

        logger.info("insight_aggregator_initialized")

    def get_daily_insights(self, date: str) -> Dict[str, Any]:
        """
        Get comprehensive daily insights.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Dict with all insights aggregated
        """
        logger.info("aggregating_daily_insights", date=date)

        # 1. Learning progress
        learning = self.learning_tracker.get_daily_summary(date)

        # 2. Shadow trade performance
        shadow_stats = self._get_shadow_trade_stats(date)

        # 3. Model improvements
        model_insights = self._get_model_insights(date)

        # 4. Gate performance
        gate_insights = self._get_gate_insights(date)

        # 5. Recommendations
        recommendations = self._generate_recommendations(
            learning, shadow_stats, model_insights, gate_insights
        )

        # 6. Overall summary
        summary = self._generate_summary(
            learning, shadow_stats, model_insights, recommendations
        )

        return {
            "date": date,
            "summary": summary,
            "learning": learning,
            "shadow_trading": shadow_stats,
            "models": model_insights,
            "gates": gate_insights,
            "recommendations": recommendations,
            "hamilton_ready": model_insights.get("ready_for_hamilton", False)
        }

    def _get_shadow_trade_stats(self, date: str) -> Dict[str, Any]:
        """Get shadow trade statistics"""
        try:
            stats = self.trade_journal.get_stats(days=1)
            return {
                "total_trades": stats.get("total_trades", 0),
                "win_rate": stats.get("win_rate", 0.0),
                "avg_pnl_bps": stats.get("avg_return_bps", 0.0),
                "note": "SIMULATED trades (paper only, no real money)"
            }
        except Exception as e:
            logger.warning("failed_to_get_shadow_stats", error=str(e))
            return {"error": str(e)}

    def _get_model_insights(self, date: str) -> Dict[str, Any]:
        """Get model improvement insights"""
        try:
            # Get improvement report
            report = self.model_evolution.get_improvement_report(days=7)

            # Check latest model readiness
            latest_model_id = report.get("last_model", {}).get("model_id")
            ready = False
            if latest_model_id:
                readiness = self.model_evolution.is_ready_for_hamilton(latest_model_id)
                ready = readiness.ready

            return {
                "models_trained_7d": report.get("models_trained", 0),
                "trend": report.get("trend", "Unknown"),
                "improvement_auc": report.get("improvement", {}).get("delta_auc", 0),
                "improvement_pct": report.get("improvement", {}).get("pct_improvement_auc", 0),
                "ready_for_hamilton": ready,
                "latest_model": report.get("last_model", {})
            }
        except Exception as e:
            logger.warning("failed_to_get_model_insights", error=str(e))
            return {"error": str(e)}

    def _get_gate_insights(self, date: str) -> Dict[str, Any]:
        """Get gate performance insights"""
        try:
            metrics = self.metrics_computer.compute_daily_metrics(date)
            gate_metrics = metrics.get("gates", {})

            return {
                "gates": gate_metrics.get("gates", []),
                "note": gate_metrics.get("note", "")
            }
        except Exception as e:
            logger.warning("failed_to_get_gate_insights", error=str(e))
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        learning: Dict,
        shadow_stats: Dict,
        model_insights: Dict,
        gate_insights: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Check shadow trading activity
        shadow_trades = shadow_stats.get("total_trades", 0)
        if shadow_trades == 0:
            recommendations.append(
                "⚠️ NO SHADOW TRADES - Gates are too strict. "
                "Consider lowering meta_label threshold from 0.45 to 0.40 for scalp mode."
            )
        elif shadow_trades < 20:
            recommendations.append(
                "📊 Low shadow trading volume - Consider loosening gate thresholds to increase learning data."
            )

        # Check learning progress
        training_sessions = learning.get("num_sessions", 0)
        if training_sessions == 0:
            recommendations.append(
                "🎓 No training sessions today - Schedule daily training at 00:00 UTC."
            )

        # Check model improvements
        if model_insights.get("trend") == "📉 Regressing":
            recommendations.append(
                "🚨 MODEL REGRESSION DETECTED - Investigate why performance dropped. Do not export to Hamilton."
            )
        elif model_insights.get("improvement_auc", 0) > 0.02:
            recommendations.append(
                f"✅ Significant improvement (+{model_insights['improvement_pct']:.1f}%) - "
                "Consider exporting to Hamilton."
            )

        # Check Hamilton readiness
        if model_insights.get("ready_for_hamilton"):
            recommendations.append(
                "🎯 Model ready for Hamilton export - Performance metrics meet all criteria."
            )
        else:
            recommendations.append(
                "⏳ Model not ready for Hamilton - Continue training to improve metrics."
            )

        # Check win rate
        win_rate = shadow_stats.get("win_rate", 0)
        if shadow_trades > 0 and win_rate < 0.70:
            recommendations.append(
                f"📉 Shadow win rate {win_rate:.1%} below target 70% - "
                "Review feature engineering and model architecture."
            )

        if not recommendations:
            recommendations.append("✓ All systems operating normally")

        return recommendations

    def _generate_summary(
        self,
        learning: Dict,
        shadow_stats: Dict,
        model_insights: Dict,
        recommendations: List[str]
    ) -> str:
        """Generate human-readable summary"""
        shadow_trades = shadow_stats.get("total_trades", 0)
        training_sessions = learning.get("num_sessions", 0)
        trend = model_insights.get("trend", "Unknown")
        improvement = model_insights.get("improvement_auc", 0)

        if shadow_trades == 0:
            summary = (
                f"⚠️ NO LEARNING TODAY - Engine received signals but gates blocked all trades. "
                f"Training sessions: {training_sessions}. "
                f"Model trend: {trend}. "
                f"Action needed: Loosen gate thresholds to enable shadow trading."
            )
        elif shadow_trades < 20:
            summary = (
                f"📊 LIMITED LEARNING - {shadow_trades} shadow trades executed. "
                f"Win rate: {shadow_stats.get('win_rate', 0):.1%}. "
                f"Training sessions: {training_sessions}. "
                f"Model trend: {trend} ({improvement:+.3f} AUC). "
                f"Action: Increase shadow trading volume for better learning."
            )
        else:
            summary = (
                f"✅ ACTIVE LEARNING - {shadow_trades} shadow trades, "
                f"win rate {shadow_stats.get('win_rate', 0):.1%}. "
                f"Training sessions: {training_sessions}. "
                f"Model trend: {trend} ({improvement:+.3f} AUC). "
                f"Hamilton ready: {model_insights.get('ready_for_hamilton', False)}."
            )

        return summary


if __name__ == '__main__':
    # Example usage
    print("Insight Aggregator Example")
    print("=" * 80)

    aggregator = InsightAggregator()

    # Get daily insights
    date = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"\n📊 Aggregating insights for {date}...")

    insights = aggregator.get_daily_insights(date)

    # Display summary
    print(f"\n{insights['summary']}")

    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")

    # Learning
    learning = insights['learning']
    print(f"\n🎓 Learning Progress:")
    print(f"  Training sessions: {learning.get('num_sessions', 0)}")
    print(f"  Samples processed: {learning.get('total_samples', 0)}")
    if learning.get('best_metrics'):
        print(f"  Best AUC: {learning['best_metrics'].get('auc', 0):.3f}")

    # Shadow trading
    shadow = insights['shadow_trading']
    print(f"\n📈 Shadow Trading:")
    print(f"  Total trades: {shadow.get('total_trades', 0)}")
    print(f"  Win rate: {shadow.get('win_rate', 0):.1%}")

    # Models
    models = insights['models']
    print(f"\n🤖 Models:")
    print(f"  Trend: {models.get('trend', 'Unknown')}")
    print(f"  Hamilton ready: {models.get('ready_for_hamilton', False)}")

    print("\n✓ Insight aggregator ready!")
