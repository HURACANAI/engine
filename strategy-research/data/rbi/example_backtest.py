"""
Example Backtest - RSI Reversal Strategy

This is a sample backtest that can be used to test the Strategy Translator
without needing to run the full RBI Agent.

Strategy:
- Buy when RSI drops below 30 (oversold)
- Sell when RSI rises above 70 (overbought)
- 15-minute timeframe
"""

from backtesting import Backtest, Strategy
import pandas as pd


class RSIReversalStrategy(Strategy):
    """
    Simple RSI reversal strategy.

    Buys when RSI < 30, sells when RSI > 70.
    """

    # Parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70

    def init(self):
        """Initialize indicators"""
        # Calculate RSI using pandas
        close = pd.Series(self.data.Close)
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        self.rsi = 100 - (100 / (1 + rs))

    def next(self):
        """Execute strategy logic on each bar"""
        # Get current RSI value
        current_rsi = self.rsi.iloc[-1]

        # Entry signal: RSI < 30 (oversold)
        if current_rsi < self.rsi_oversold:
            if not self.position:
                self.buy()

        # Exit signal: RSI > 70 (overbought)
        elif current_rsi > self.rsi_overbought:
            if self.position:
                self.sell()


# Backtest configuration
if __name__ == "__main__":
    # Load data (example - would normally load real data)
    # df = pd.read_csv('BTC-USD-15m.csv', parse_dates=['timestamp'])

    # Example usage:
    # bt = Backtest(df, RSIReversalStrategy, cash=10000, commission=0.002)
    # stats = bt.run()
    # print(stats)

    print("RSI Reversal Strategy - Example Backtest")
    print("Entry: RSI < 30")
    print("Exit: RSI > 70")
    print("Timeframe: 15m")
