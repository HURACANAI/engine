"""Notification service for Telegram and monitoring alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Any, Iterable, Sequence, Optional

import requests
import structlog

from ..config.settings import NotificationSettings


logger = structlog.get_logger(__name__)


@dataclass
class TelegramMessage:
    text: str


class NotificationClient:
    """Dispatches success, rejection, and summary messages to Telegram."""

    def __init__(self, settings: NotificationSettings) -> None:
        self._settings = settings

    def _generate_grok_explanation(self, metrics: dict, symbol: str, date: str, costs_bps: float, validation_window: Optional[str] = None, num_splits: Optional[int] = None) -> Optional[str]:
        """Generate an AI explanation of the metrics using Grok API."""
        if not self._settings.grok_enabled:
            logger.debug("grok_disabled", message="Grok explanations are disabled")
            return None
            
        if not self._settings.grok_api_key:
            logger.warning("grok_api_key_missing", message="Grok API key not configured")
            return None
        
        try:
            # Trim and validate API key
            api_key = (self._settings.grok_api_key or "").strip()
            if not api_key or len(api_key) < 10:
                logger.warning("grok_api_key_invalid", message="API key is empty or too short", key_length=len(api_key) if api_key else 0)
                return None
            
            # Log key prefix for debugging (first 10 chars only)
            logger.debug("grok_api_call_starting", api_key_prefix=api_key[:10] + "..." if len(api_key) > 10 else "***", key_length=len(api_key))
            
            sharpe = metrics.get("sharpe", 0.0)
            profit_factor = metrics.get("profit_factor", 0.0)
            hit_rate = metrics.get("hit_rate", 0.0) * 100
            max_dd = metrics.get("max_dd_bps", 0.0)
            edge = metrics.get("recommended_edge_threshold_bps", 0)
            trades = metrics.get("trades_oos", 0)
            
            # Build dataset and testing info
            dataset_info = ""
            if validation_window:
                dataset_info = f"\nTraining Dataset: {validation_window}"
            if num_splits:
                dataset_info += f"\nWalk-Forward Tests: Model was tested {num_splits} times on different time periods"
            
            prompt = f"""Analyze these trading performance metrics and provide a simple, comprehensive explanation:

Trading Pair: {symbol}
Model Date: {date}{dataset_info}
Sharpe Ratio: {sharpe:.2f}
Profit Factor: {profit_factor:.2f}
Win Rate (Hit Rate): {hit_rate:.1f}%
Max Drawdown: {max_dd:.1f} basis points
Trading Costs: {costs_bps:.1f} basis points
Edge Threshold: {edge} basis points
Total Trades Tested: {trades}

Please provide a clear analysis that includes:
1. TESTING ROBUSTNESS: How many times was this model tested? What does testing on {num_splits if num_splits else 'multiple'} different time periods tell us about reliability?
2. DATASET ANALYSIS: What period was the model trained on? Does this give us confidence in future performance?
3. PROFIT ANALYSIS: What do these numbers mean in terms of actual profitability? Is this strategy profitable after costs?
4. WIN RATE ANALYSIS: How reliable is the win rate? What does {hit_rate:.1f}% win rate mean in practical terms?
5. RISK ASSESSMENT: Is the max drawdown acceptable? What does {max_dd:.1f} bps drawdown mean?
6. OVERALL VERDICT: Is this a good strategy to use? Should we be confident, cautious, or concerned?

Keep the explanation simple and practical - focus on what these numbers mean for making money, not technical jargon."""

            # Verify API key format (should start with gsk_)
            if not api_key.startswith("gsk_"):
                logger.warning(
                    "grok_api_key_format_invalid",
                    message="API key should start with 'gsk_'",
                    key_prefix=api_key[:10] + "..." if len(api_key) > 10 else "***"
                )
                return None
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-latest",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert trading analyst. Explain trading metrics in simple, practical terms focused on profitability, win rates, and risk. Be clear and actionable."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 400
                },
                timeout=15
            )
            
            if response.ok:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    explanation = result['choices'][0]['message']['content'].strip()
                    logger.info("grok_explanation_generated", symbol=symbol, explanation_length=len(explanation))
                    return explanation
                else:
                    logger.warning("grok_api_invalid_response", response=result)
                    return None
            else:
                error_text = response.text[:500] if response.text else "No error message"
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_text) if isinstance(error_json, dict) else error_text
                except:
                    error_detail = error_text
                
                logger.warning(
                    "grok_api_failed",
                    status=response.status_code,
                    error=error_detail,
                    api_key_prefix=api_key[:10] + "..." if len(api_key) > 10 else "***",
                    key_length=len(api_key),
                    key_starts_with_gsk=api_key.startswith("gsk_")
                )
                return None
                
        except Exception as e:
            logger.warning("grok_explanation_error", error=str(e), error_type=type(e).__name__)
            return None

    def send_success(self, result: Any, run_date: date) -> None:
        profit_factor = result.metrics.get("profit_factor", 0.0)
        # Format profit factor: cap at 999.99 for display
        pf_display = min(profit_factor, 999.99)
        pf_str = f"{pf_display:.2f}" if pf_display < 999.99 else "999.99+"
        
        message = (
            "✅ Baseline {symbol} {date} | Sharpe {sharpe:.2f} | PF {profit_factor} | "
            "Hit {hit_rate:.1f}% | MaxDD {max_dd:.1f} bps | Costs {costs:.1f} bps | "
            "EdgeThr {edge} bps | OOS trades {trades}"
        ).format(
            symbol=result.symbol,
            date=run_date.isoformat(),
            sharpe=result.metrics.get("sharpe", 0.0),
            profit_factor=pf_str,
            hit_rate=result.metrics.get("hit_rate", 0.0) * 100,
            max_dd=result.metrics.get("max_dd_bps", 0.0),
            costs=result.costs.total_costs_bps,
            edge=result.metrics.get("recommended_edge_threshold_bps", 0),
            trades=result.metrics.get("trades_oos", 0),
        )
        
        # Extract validation window and split count from metrics
        validation_window = result.metrics.get("validation_window")
        num_splits = result.metrics.get("num_walk_forward_splits")
        # Try to extract from metrics_payload if not in metrics
        if hasattr(result, 'metrics_payload') and result.metrics_payload:
            if not validation_window and hasattr(result.metrics_payload, 'validation_window'):
                validation_window = result.metrics_payload.validation_window
        
        # Generate AI explanation using Grok
        explanation = self._generate_grok_explanation(
            metrics=result.metrics,
            symbol=result.symbol,
            date=run_date.isoformat(),
            costs_bps=result.costs.total_costs_bps,
            validation_window=validation_window,
            num_splits=num_splits
        )
        
        # Append explanation below the original output
        if explanation:
            message += f"\n\n💡 AI Analysis:\n\n{explanation}"
        
        self._send(message)

    def send_reject(self, result: Any, run_date: date) -> None:
        message = (
            "❌ Baseline {symbol} {date} rejected → kept prior. Reason: {reason}. Costs {costs:.1f} bps"
        ).format(
            symbol=result.symbol,
            date=run_date.isoformat(),
            reason=result.reason,
            costs=result.costs.total_costs_bps,
        )
        self._send(message)

    def send_summary(self, results: Sequence[Any], run_date: date) -> None:
        result_list = list(results)
        promoted = sum(1 for r in result_list if r.published)
        rejected = sum(1 for r in result_list if not r.published)
        total = len(result_list)
        sharpe_values = [r.metrics.get("sharpe", 0.0) for r in result_list if r.metrics.get("sharpe") is not None]
        cost_values = [r.costs.total_costs_bps for r in result_list]
        median_sharpe = median(sharpe_values) if sharpe_values else 0.0
        median_costs = median(cost_values) if cost_values else 0.0
        message = (
            "🧠 {date} Baselines built: {built}/{total} | Promoted: {promoted} | Rejected: {rejected} | "
            "Median Sharpe {median_sharpe:.2f} | Median Costs {median_costs:.1f} bps"
        ).format(
            date=run_date.isoformat(),
            built=total,
            total=total,
            promoted=promoted,
            rejected=rejected,
            median_sharpe=median_sharpe,
            median_costs=median_costs,
        )
        self._send(message)

    def _send(self, message: str) -> None:
        if not self._settings.telegram_enabled:
            logger.info("notification_skipped", message=message)
            return
        if not self._settings.telegram_webhook_url or not self._settings.telegram_chat_id:
            logger.warning("telegram_missing_config", message=message)
            return
        payload = {
            "chat_id": self._settings.telegram_chat_id,
            "text": message,
        }
        response = requests.post(self._settings.telegram_webhook_url, json=payload, timeout=10)
        if not response.ok:
            logger.error("telegram_send_failed", status=response.status_code, text=response.text)

