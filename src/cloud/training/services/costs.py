"""Cost modeling service for fees, spreads, and slippage estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..config.settings import CostSettings


@dataclass(frozen=True)
class CostBreakdown:
    fee_bps: float
    spread_bps: float
    slippage_bps: float

    @property
    def total_costs_bps(self) -> float:
        return self.fee_bps + self.spread_bps + self.slippage_bps


class CostModel:
    """Computes per-coin cost components used in labeling and metrics."""

    def __init__(self, settings: CostSettings) -> None:
        self._settings = settings

    def estimate(
        self,
        *,
        taker_fee_bps: float | None,
        spread_bps: float | None,
        volatility_bps: float,
        adv_quote: float | None,
    ) -> CostBreakdown:
        fee = taker_fee_bps if taker_fee_bps is not None else self._settings.default_fee_bps
        spread = spread_bps if spread_bps is not None else self._settings.default_spread_bps

        notionals_ratio = 0.0
        if adv_quote and adv_quote > 0:
            notionals_ratio = self._settings.notional_per_trade / adv_quote

        # Base slippage anchored to volatility * participation
        slip = self._settings.slippage_alpha * volatility_bps * notionals_ratio

        # Floor to avoid zero slippage environments
        slip = max(slip, self._settings.slippage_floor_bps)

        # Penalise thin liquidity buckets (participation ratio is higher)
        if notionals_ratio > 0:
            penalties = self._settings.adv_penalties_bps
            breakpoints = self._settings.adv_liquidity_breakpoints
            for idx, threshold in enumerate(breakpoints):
                if notionals_ratio >= threshold:
                    slip += penalties[min(idx, len(penalties) - 1)]
                else:
                    break

        # Volatility shock multiplier (helps account for regime transitions)
        slip += self._settings.volatility_slippage_multiplier * volatility_bps

        return CostBreakdown(fee_bps=fee, spread_bps=spread, slippage_bps=slip)

    def recommended_edge_threshold(self, costs: CostBreakdown) -> int:
        gross = costs.total_costs_bps + self._settings.target_net_bps + self._settings.taker_buffer_bps
        return int(round(gross))
