from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


@dataclass
class WindowSearchResult:
    window_days: int
    score: float
    volatility: float
    splits_estimate: int
    coverage_days: float


class TrainingWindowSearcher:
    """Lightweight heuristic searcher for selecting a training window length."""

    def __init__(
        self,
        candidate_windows: List[int],
        train_days: int,
        test_days: int,
    ) -> None:
        self.candidate_windows = sorted(set(candidate_windows), reverse=True)
        self.train_days = train_days
        self.test_days = test_days

    def select_best_window(self, frame: pl.DataFrame) -> WindowSearchResult:
        if frame.is_empty() or "ts" not in frame.columns:
            raise ValueError("Cannot run window search without timestamp data")

        ts_max = frame["ts"].max()
        best_result: WindowSearchResult | None = None

        for window in self.candidate_windows:
            window_start = ts_max - timedelta(days=window)
            window_frame = frame.filter(pl.col("ts") >= window_start)
            if window_frame.is_empty():
                continue

            coverage_days = (ts_max - window_frame["ts"].min()).total_seconds() / 86400.0
            approx_splits = max(
                int((coverage_days - (self.train_days + self.test_days)) // self.test_days), 0
            )
            volatility = self._estimate_volatility(window_frame)
            score = (approx_splits + 1) * (1 + volatility)

            logger.debug(
                "window_search_candidate_scored",
                window_days=window,
                score=score,
                volatility=volatility,
                splits_estimate=approx_splits,
                coverage_days=coverage_days,
            )

            candidate = WindowSearchResult(
                window_days=window,
                score=score,
                volatility=volatility,
                splits_estimate=approx_splits,
                coverage_days=coverage_days,
            )

            if best_result is None or candidate.score > best_result.score:
                best_result = candidate

        if best_result is None:
            raise ValueError("Unable to determine suitable training window")

        logger.info(
            "window_search_selected",
            window_days=best_result.window_days,
            score=best_result.score,
            splits_estimate=best_result.splits_estimate,
            volatility=best_result.volatility,
        )
        return best_result

    @staticmethod
    def _estimate_volatility(window_frame: pl.DataFrame) -> float:
        if "close" not in window_frame.columns:
            return 0.0
        closes = window_frame["close"].to_numpy()
        if closes.size < 2:
            return 0.0
        returns = np.diff(closes) / closes[:-1]
        vol = float(np.nanstd(returns))
        return max(vol, 0.0)


