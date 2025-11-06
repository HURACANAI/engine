"""
Historical Trade Exporter

Exports historical trades from your database/logs with all necessary fields
for gate calibration and meta-label training.

Required Fields:
- Signal: technique, confidence, regime, edge_hat_bps
- Features: Dict of market features at signal time
- Execution: order_type, spread_bps, liquidity_score
- Outcome: won (bool), pnl_bps, hold_time_sec
- Metadata: timestamp, symbol

Usage:
    exporter = TradeExporter(
        data_source='database',  # or 'csv', 'logs'
        connection_string='postgresql://...',
    )

    # Export last 3 months
    trades = exporter.export_trades(
        start_date='2024-08-01',
        end_date='2024-11-01',
        min_confidence=0.50,  # Filter low-quality signals
    )

    # Save for calibration
    exporter.save_for_calibration(trades, 'historical_trades.pkl')
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pickle
import json
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradeExport:
    """Exported trade record."""

    # Signal
    technique: str
    confidence: float
    regime: str
    edge_hat_bps: float

    # Features
    features: Dict[str, float]

    # Execution
    order_type: str
    spread_bps: float
    liquidity_score: float

    # Outcome
    won: bool
    pnl_bps: float
    hold_time_sec: float

    # Metadata
    timestamp: float
    symbol: str
    trade_id: Optional[str] = None


class TradeExporter:
    """
    Export historical trades for calibration.

    Supports multiple data sources:
    1. Database (PostgreSQL, MySQL)
    2. CSV files
    3. Log files (structured JSON logs)
    4. In-memory (for testing)
    """

    def __init__(
        self,
        data_source: str = 'database',
        connection_string: Optional[str] = None,
        csv_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):
        """
        Initialize trade exporter.

        Args:
            data_source: 'database', 'csv', 'logs', or 'memory'
            connection_string: Database connection string
            csv_path: Path to CSV file
            log_path: Path to log files
        """
        self.data_source = data_source
        self.connection_string = connection_string
        self.csv_path = csv_path
        self.log_path = log_path

        logger.info(
            "trade_exporter_initialized",
            data_source=data_source,
        )

    def export_trades(
        self,
        start_date: str,
        end_date: str,
        min_confidence: float = 0.50,
        symbols: Optional[List[str]] = None,
    ) -> List[TradeExport]:
        """
        Export trades from data source.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_confidence: Minimum confidence to include
            symbols: List of symbols to include (None = all)

        Returns:
            List of TradeExport objects
        """
        if self.data_source == 'database':
            return self._export_from_database(start_date, end_date, min_confidence, symbols)
        elif self.data_source == 'csv':
            return self._export_from_csv(start_date, end_date, min_confidence, symbols)
        elif self.data_source == 'logs':
            return self._export_from_logs(start_date, end_date, min_confidence, symbols)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

    def _export_from_database(
        self,
        start_date: str,
        end_date: str,
        min_confidence: float,
        symbols: Optional[List[str]],
    ) -> List[TradeExport]:
        """Export from database."""
        # TODO: Implement database export
        # This is a template - customize for your database schema

        logger.warning("database_export_not_implemented")

        # Example SQL query structure:
        """
        SELECT
            t.technique,
            t.confidence,
            t.regime,
            t.edge_hat_bps,
            t.features,  -- JSON column
            t.order_type,
            t.spread_bps,
            t.liquidity_score,
            t.won,
            t.pnl_bps,
            t.hold_time_sec,
            t.timestamp,
            t.symbol,
            t.trade_id
        FROM trades t
        WHERE t.timestamp >= %s
          AND t.timestamp <= %s
          AND t.confidence >= %s
          AND (t.symbol = ANY(%s) OR %s IS NULL)
        ORDER BY t.timestamp ASC
        """

        return []

    def _export_from_csv(
        self,
        start_date: str,
        end_date: str,
        min_confidence: float,
        symbols: Optional[List[str]],
    ) -> List[TradeExport]:
        """Export from CSV file."""
        if not self.csv_path:
            raise ValueError("csv_path required for CSV export")

        logger.info("exporting_from_csv", path=self.csv_path)

        # Read CSV
        df = pd.read_csv(self.csv_path)

        # Parse dates
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter
        mask = (
            (df['timestamp'] >= start_dt) &
            (df['timestamp'] <= end_dt) &
            (df['confidence'] >= min_confidence)
        )

        if symbols:
            mask &= df['symbol'].isin(symbols)

        df_filtered = df[mask]

        # Convert to TradeExport objects
        trades = []
        for _, row in df_filtered.iterrows():
            # Parse features JSON
            if isinstance(row['features'], str):
                features = json.loads(row['features'])
            else:
                features = row['features']

            trade = TradeExport(
                technique=row['technique'],
                confidence=float(row['confidence']),
                regime=row['regime'],
                edge_hat_bps=float(row['edge_hat_bps']),
                features=features,
                order_type=row['order_type'],
                spread_bps=float(row['spread_bps']),
                liquidity_score=float(row['liquidity_score']),
                won=bool(row['won']),
                pnl_bps=float(row['pnl_bps']),
                hold_time_sec=float(row['hold_time_sec']),
                timestamp=row['timestamp'].timestamp(),
                symbol=row['symbol'],
                trade_id=row.get('trade_id'),
            )
            trades.append(trade)

        logger.info("csv_export_complete", trades=len(trades))

        return trades

    def _export_from_logs(
        self,
        start_date: str,
        end_date: str,
        min_confidence: float,
        symbols: Optional[List[str]],
    ) -> List[TradeExport]:
        """Export from structured log files."""
        # TODO: Implement log parsing
        logger.warning("log_export_not_implemented")
        return []

    def save_for_calibration(
        self,
        trades: List[TradeExport],
        output_path: str,
    ) -> None:
        """
        Save trades for calibration.

        Args:
            trades: List of exported trades
            output_path: Path to save (.pkl or .csv)
        """
        if output_path.endswith('.pkl'):
            # Save as pickle (preserves types)
            with open(output_path, 'wb') as f:
                pickle.dump(trades, f)

            logger.info("trades_saved_pickle", path=output_path, count=len(trades))

        elif output_path.endswith('.csv'):
            # Save as CSV (for inspection)
            data = []
            for trade in trades:
                trade_dict = asdict(trade)
                trade_dict['features'] = json.dumps(trade_dict['features'])
                data.append(trade_dict)

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

            logger.info("trades_saved_csv", path=output_path, count=len(trades))

        else:
            raise ValueError(f"Unsupported file type: {output_path}")

    def load_for_calibration(
        self,
        input_path: str,
    ) -> List[TradeExport]:
        """
        Load trades from saved file.

        Args:
            input_path: Path to load (.pkl or .csv)

        Returns:
            List of TradeExport objects
        """
        if input_path.endswith('.pkl'):
            with open(input_path, 'rb') as f:
                trades = pickle.load(f)

            logger.info("trades_loaded_pickle", path=input_path, count=len(trades))

        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)

            trades = []
            for _, row in df.iterrows():
                features = json.loads(row['features'])

                trade = TradeExport(
                    technique=row['technique'],
                    confidence=float(row['confidence']),
                    regime=row['regime'],
                    edge_hat_bps=float(row['edge_hat_bps']),
                    features=features,
                    order_type=row['order_type'],
                    spread_bps=float(row['spread_bps']),
                    liquidity_score=float(row['liquidity_score']),
                    won=bool(row['won']),
                    pnl_bps=float(row['pnl_bps']),
                    hold_time_sec=float(row['hold_time_sec']),
                    timestamp=float(row['timestamp']),
                    symbol=row['symbol'],
                    trade_id=row.get('trade_id'),
                )
                trades.append(trade)

            logger.info("trades_loaded_csv", path=input_path, count=len(trades))

        else:
            raise ValueError(f"Unsupported file type: {input_path}")

        return trades

    def get_summary_stats(
        self,
        trades: List[TradeExport],
    ) -> Dict:
        """Get summary statistics of exported trades."""
        if not trades:
            return {}

        wins = sum(1 for t in trades if t.won)
        total = len(trades)

        # By technique
        by_technique = {}
        for trade in trades:
            tech = trade.technique
            if tech not in by_technique:
                by_technique[tech] = {'total': 0, 'wins': 0}
            by_technique[tech]['total'] += 1
            if trade.won:
                by_technique[tech]['wins'] += 1

        # By regime
        by_regime = {}
        for trade in trades:
            regime = trade.regime
            if regime not in by_regime:
                by_regime[regime] = {'total': 0, 'wins': 0}
            by_regime[regime]['total'] += 1
            if trade.won:
                by_regime[regime]['wins'] += 1

        # P&L stats
        all_pnl = [t.pnl_bps for t in trades]
        winning_pnl = [t.pnl_bps for t in trades if t.won]
        losing_pnl = [t.pnl_bps for t in trades if not t.won]

        return {
            'total_trades': total,
            'win_rate': wins / total,
            'avg_pnl_bps': sum(all_pnl) / total,
            'avg_win_bps': sum(winning_pnl) / len(winning_pnl) if winning_pnl else 0,
            'avg_loss_bps': sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0,
            'by_technique': {
                tech: {
                    'count': stats['total'],
                    'win_rate': stats['wins'] / stats['total'],
                }
                for tech, stats in by_technique.items()
            },
            'by_regime': {
                regime: {
                    'count': stats['total'],
                    'win_rate': stats['wins'] / stats['total'],
                }
                for regime, stats in by_regime.items()
            },
        }


def create_template_csv(output_path: str = 'trade_template.csv'):
    """Create a template CSV file showing required format."""
    template_data = {
        'technique': ['TREND', 'RANGE', 'BREAKOUT'],
        'confidence': [0.75, 0.60, 0.82],
        'regime': ['TREND', 'RANGE', 'TREND'],
        'edge_hat_bps': [15.0, 10.0, 20.0],
        'features': [
            '{"trend_strength": 0.8, "adx": 35}',
            '{"compression": 0.7, "bb_width": 0.02}',
            '{"breakout_strength": 0.85}',
        ],
        'order_type': ['maker', 'maker', 'taker'],
        'spread_bps': [8.0, 6.0, 10.0],
        'liquidity_score': [0.80, 0.75, 0.70],
        'won': [True, True, False],
        'pnl_bps': [120.0, 90.0, -50.0],
        'hold_time_sec': [45.0, 15.0, 120.0],
        'timestamp': [1699000000.0, 1699000060.0, 1699000120.0],
        'symbol': ['ETH-USD', 'ETH-USD', 'BTC-USD'],
        'trade_id': ['trade_001', 'trade_002', 'trade_003'],
    }

    df = pd.DataFrame(template_data)
    df.to_csv(output_path, index=False)

    print(f"Template CSV created: {output_path}")
    print("\nFormat your historical trades as CSV with these columns:")
    print(df.columns.tolist())


if __name__ == '__main__':
    # Create template
    create_template_csv()

    print("\n" + "=" * 70)
    print("TRADE EXPORT INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Export your historical trades to CSV format (see trade_template.csv)
2. Include all required columns
3. Features should be JSON string: '{"feature1": value1, "feature2": value2}'
4. Save to: historical_trades.csv

Then run:
    exporter = TradeExporter(data_source='csv', csv_path='historical_trades.csv')
    trades = exporter.export_trades(
        start_date='2024-08-01',
        end_date='2024-11-01',
    )
    exporter.save_for_calibration(trades, 'trades_for_calibration.pkl')
    """)
