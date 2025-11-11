"""
Scalable Engine Usage Example

Demonstrates how to use the scalable engine for 400 coins and 500 trades.

Author: Huracan Engine Team
Date: 2025-01-27
"""

import asyncio
import yaml
from pathlib import Path

from src.cloud.training.infrastructure.scalable_engine import (
    ScalableEngine,
    ScalableEngineConfig,
    create_scalable_engine_from_config,
)
from src.cloud.training.infrastructure import Message, StreamType


async def main():
    """Main function demonstrating scalable engine usage."""
    
    # Load configuration
    config_path = Path("engine/config/base.yaml")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Create engine from config
    engine = create_scalable_engine_from_config(config_dict.get("engine", {}))
    
    # Initialize engine
    await engine.initialize()
    
    # Define coins to trade (can be up to 400)
    coins = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "MATIC", "AVAX",
        "LTC", "LINK", "UNI", "ATOM", "ETC", "XLM", "BCH", "FIL", "TRX", "ETC",
        # Add more coins up to 400...
    ]
    
    # Start engine with active coins (start with 20, scale up to 400)
    await engine.start(coins[:engine.config.active_coins])
    
    # Run health checks
    health_status = await engine.get_health_status()
    print(f"Health Status: {health_status['status']}")
    for check_name, (healthy, message) in health_status['checks'].items():
        print(f"  {check_name}: {'✓' if healthy else '✗'} {message}")
    
    # Get metrics
    metrics = engine.get_metrics()
    print(f"\nMetrics:")
    print(f"  Active Coins: {metrics['engine']['active_coins']}")
    print(f"  Running: {metrics['engine']['running']}")
    print(f"  Active Trades: {metrics['risk']['active_trades']}")
    print(f"  Global Exposure: {metrics['risk']['global']['total_exposure_pct']:.2f}%")
    
    # Simulate market data (in production, this would come from exchange WebSockets)
    message_bus = engine.message_bus
    
    # Publish sample market data
    for coin in coins[:engine.config.active_coins]:
        await message_bus.publish(Message(
            stream_type=StreamType.MARKET_DATA,
            coin=coin,
            data={
                "price": 50000.0,
                "volume": 1000.0,
                "timestamp": asyncio.get_event_loop().time(),
            },
            timestamp=asyncio.get_event_loop().time(),
        ))
    
    # Run for a while (in production, this would run indefinitely)
    print("\nEngine running... (Press Ctrl+C to stop)")
    try:
        await asyncio.sleep(60)  # Run for 60 seconds
    except KeyboardInterrupt:
        print("\nStopping engine...")
    finally:
        # Stop engine
        await engine.stop()
        print("Engine stopped")


if __name__ == "__main__":
    asyncio.run(main())

