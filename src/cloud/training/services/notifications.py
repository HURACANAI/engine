"""Notification service for Telegram and monitoring alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Any, Iterable, Sequence

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

    def send_success(self, result: Any, run_date: date) -> None:
        message = (
            "✅ Baseline {symbol} {date} | Sharpe {sharpe:.2f} | PF {profit_factor:.2f} | "
            "Hit {hit_rate:.1f}% | MaxDD {max_dd:.1f} bps | Costs {costs:.1f} bps | "
            "EdgeThr {edge} bps | OOS trades {trades}"
        ).format(
            symbol=result.symbol,
            date=run_date.isoformat(),
            sharpe=result.metrics.get("sharpe", 0.0),
            profit_factor=result.metrics.get("profit_factor", 0.0),
            hit_rate=result.metrics.get("hit_rate", 0.0) * 100,
            max_dd=result.metrics.get("max_dd_bps", 0.0),
            costs=result.costs.total_costs_bps,
            edge=result.metrics.get("recommended_edge_threshold_bps", 0),
            trades=result.metrics.get("trades_oos", 0),
        )
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

