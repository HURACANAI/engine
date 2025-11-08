"""Comprehensive model evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class ComprehensiveEvaluation:
    """
    Comprehensive model evaluation with all key metrics:
    - Sharpe ratio
    - Sortino ratio
    - Hit ratio
    - Profit factor
    - Max drawdown
    - Calmar ratio
    - Win rate
    - Average win/loss
    - Expectancy
    - Risk-adjusted returns
    """

    def __init__(
        self,
        brain_library: Optional[BrainLibrary] = None,
    ) -> None:
        """
        Initialize comprehensive evaluation.
        
        Args:
            brain_library: Optional Brain Library instance for storage
        """
        self.brain = brain_library
        logger.info("comprehensive_evaluation_initialized")

    def evaluate_model(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        returns: Optional[np.ndarray] = None,
        trades: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model with comprehensive metrics.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            returns: Optional returns array (for financial metrics)
            trades: Optional trades DataFrame (for trade-based metrics)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic prediction metrics
        metrics.update(self._calculate_prediction_metrics(predictions, actuals))
        
        # Financial metrics (if returns provided)
        if returns is not None:
            metrics.update(self._calculate_financial_metrics(returns))
        
        # Trade-based metrics (if trades provided)
        if trades is not None and not trades.empty:
            metrics.update(self._calculate_trade_metrics(trades))
        
        # Risk metrics
        if returns is not None:
            metrics.update(self._calculate_risk_metrics(returns))
        
        logger.info("model_evaluation_complete", num_metrics=len(metrics))
        
        return metrics

    def _calculate_prediction_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        # Mean Absolute Error
        mae = float(np.mean(np.abs(predictions - actuals)))
        
        # Mean Squared Error
        mse = float(np.mean((predictions - actuals) ** 2))
        
        # Root Mean Squared Error
        rmse = float(np.sqrt(mse))
        
        # Mean Absolute Percentage Error
        mape = float(np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100)
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        # Accuracy (within 1% tolerance)
        accuracy = float(np.mean(np.abs(predictions - actuals) < 0.01))
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "r_squared": r_squared,
            "accuracy": accuracy,
        }

    def _calculate_financial_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate financial performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Total return
        total_return = float(np.sum(returns))
        
        # Mean return
        mean_return = float(np.mean(returns))
        
        # Annualized return (assuming 252 trading days)
        annualized_return = float(mean_return * 252)
        
        # Volatility (standard deviation)
        volatility = float(np.std(returns))
        
        # Annualized volatility
        annualized_volatility = float(volatility * np.sqrt(252))
        
        # Sharpe ratio (risk-free rate assumed to be 0)
        sharpe_ratio = float(mean_return / (volatility + 1e-8) * np.sqrt(252))
        
        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        sortino_ratio = float(mean_return / (downside_std + 1e-8) * np.sqrt(252)) if downside_std > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Calmar ratio (annual return / max drawdown)
        calmar_ratio = float(annualized_return / (abs(max_drawdown) + 1e-8)) if abs(max_drawdown) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "mean_return": mean_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": abs(max_drawdown),  # Positive value
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        if trades.empty:
            return {}
        
        # Required columns
        required_cols = ["pnl_bps", "timestamp"]
        if not all(col in trades.columns for col in required_cols):
            logger.warning("missing_trade_columns", required=required_cols, available=list(trades.columns))
            return {}
        
        # Win/Loss analysis
        pnl = trades["pnl_bps"].values
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        
        # Hit ratio (win rate)
        hit_ratio = float(len(wins) / len(pnl)) if len(pnl) > 0 else 0.0
        
        # Profit factor
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = float(gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else 0.0
        
        # Average win
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        
        # Average loss
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        
        # Win/Loss ratio
        win_loss_ratio = float(avg_win / (abs(avg_loss) + 1e-8)) if abs(avg_loss) > 0 else 0.0
        
        # Expectancy (expected value per trade)
        expectancy = float((hit_ratio * avg_win) - ((1 - hit_ratio) * abs(avg_loss)))
        
        # Largest win
        largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
        
        # Largest loss
        largest_loss = float(np.min(losses)) if len(losses) > 0 else 0.0
        
        # Total trades
        total_trades = len(trades)
        
        return {
            "hit_ratio": hit_ratio,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "expectancy": expectancy,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "total_trades": float(total_trades),
        }

    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics."""
        if len(returns) == 0:
            return {}
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = float(np.percentile(returns, 5))
        
        # Conditional Value at Risk (CVaR) - Expected shortfall
        cvar_95 = float(np.mean(returns[returns <= var_95])) if len(returns[returns <= var_95]) > 0 else 0.0
        
        # Skewness
        skewness = float(self._calculate_skewness(returns))
        
        # Kurtosis
        kurtosis = float(self._calculate_kurtosis(returns))
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        
        return {
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "downside_deviation": downside_deviation,
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skew = np.mean(((data - mean) / std) ** 3)
        return float(skew)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        kurt = np.mean(((data - mean) / std) ** 4) - 3.0  # Excess kurtosis
        return float(kurt)

    def store_evaluation(
        self,
        model_id: str,
        symbol: str,
        metrics: Dict[str, float],
        evaluation_date: Optional[datetime] = None,
    ) -> None:
        """
        Store evaluation metrics in Brain Library.
        
        Args:
            model_id: Model identifier
            symbol: Trading symbol
            metrics: Evaluation metrics
            evaluation_date: Evaluation date (default: now)
        """
        if not self.brain:
            logger.warning("brain_library_not_available", message="Cannot store evaluation metrics")
            return
        
        if evaluation_date is None:
            evaluation_date = datetime.now(tz=timezone.utc)
        
        try:
            # Store model metrics
            self.brain.store_model_metrics(
                model_id=model_id,
                evaluation_date=evaluation_date,
                symbol=symbol,
                metrics={
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "sortino_ratio": metrics.get("sortino_ratio", 0.0),
                    "hit_ratio": metrics.get("hit_ratio", 0.0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "calmar_ratio": metrics.get("calmar_ratio", 0.0),
                },
            )
            
            logger.info("evaluation_metrics_stored", model_id=model_id, symbol=symbol)
        except Exception as e:
            logger.warning("evaluation_storage_failed", model_id=model_id, error=str(e))

    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
    ) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("Model Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Prediction Metrics
        report.append("Prediction Metrics:")
        report.append(f"  MAE: {metrics.get('mae', 0):.6f}")
        report.append(f"  RMSE: {metrics.get('rmse', 0):.6f}")
        report.append(f"  R²: {metrics.get('r_squared', 0):.4f}")
        report.append(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
        report.append("")
        
        # Financial Metrics
        report.append("Financial Metrics:")
        report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        report.append(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
        report.append("")
        
        # Trade Metrics
        if "hit_ratio" in metrics:
            report.append("Trade Metrics:")
            report.append(f"  Hit Ratio: {metrics.get('hit_ratio', 0):.2%}")
            report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}")
            report.append(f"  Expectancy: {metrics.get('expectancy', 0):.6f}")
            report.append(f"  Total Trades: {int(metrics.get('total_trades', 0))}")
            report.append("")
        
        # Risk Metrics
        report.append("Risk Metrics:")
        report.append(f"  VaR (95%): {metrics.get('var_95', 0):.6f}")
        report.append(f"  CVaR (95%): {metrics.get('cvar_95', 0):.6f}")
        report.append(f"  Skewness: {metrics.get('skewness', 0):.4f}")
        report.append(f"  Kurtosis: {metrics.get('kurtosis', 0):.4f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

