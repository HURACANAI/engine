#!/usr/bin/env python3
"""Quick test script for Grok API and Telegram integration."""

import sys
from pathlib import Path
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud.training.config.settings import EngineSettings
from cloud.training.services.notifications import NotificationClient
from cloud.training.services.costs import CostBreakdown


def main():
    """Test Grok API and Telegram integration."""
    print("🧪 Testing Grok API and Telegram Integration...")
    print("=" * 60)
    
    # Load settings
    print("\n1️⃣  Loading settings from config...")
    try:
        settings = EngineSettings.load()
        notification_settings = settings.notifications
        print(f"   ✅ Settings loaded")
        print(f"   📱 Telegram enabled: {notification_settings.telegram_enabled}")
        print(f"   🤖 Grok enabled: {notification_settings.grok_enabled}")
        print(f"   🔑 Grok API key: {'✅ Set' if notification_settings.grok_api_key else '❌ Not set'}")
        if notification_settings.grok_api_key:
            key_preview = notification_settings.grok_api_key[:10] + "..." if len(notification_settings.grok_api_key) > 10 else "***"
            print(f"   🔑 Key preview: {key_preview}")
    except Exception as e:
        print(f"   ❌ Failed to load settings: {e}")
        return
    
    # Create notification client
    print("\n2️⃣  Creating notification client...")
    try:
        client = NotificationClient(notification_settings)
        print("   ✅ Notification client created")
    except Exception as e:
        print(f"   ❌ Failed to create client: {e}")
        return
    
    # Create sample metrics
    print("\n3️⃣  Creating sample metrics...")
    sample_metrics = {
        "sharpe": 75.13,
        "profit_factor": 999.99,
        "hit_rate": 1.0,  # 100%
        "max_dd_bps": -21984.87,
        "recommended_edge_threshold_bps": 37,
        "trades_oos": 2607,
    }
    
    # Create mock result object
    class MockResult:
        def __init__(self):
            self.symbol = "SOL/USDT"
            self.metrics = sample_metrics
            self.costs = CostBreakdown(
                fee_bps=8.0,
                spread_bps=6.0,
                slippage_bps=0.0,
            )
    
    mock_result = MockResult()
    print("   ✅ Sample metrics created")
    print(f"      Symbol: {mock_result.symbol}")
    print(f"      Sharpe: {sample_metrics['sharpe']:.2f}")
    print(f"      Profit Factor: {sample_metrics['profit_factor']:.2f}")
    print(f"      Hit Rate: {sample_metrics['hit_rate']*100:.1f}%")
    
    # Test Grok explanation
    print("\n4️⃣  Testing Grok API explanation generation...")
    try:
        explanation = client._generate_grok_explanation(
            metrics=sample_metrics,
            symbol=mock_result.symbol,
            date=date.today().isoformat(),
            costs_bps=mock_result.costs.total_costs_bps
        )
        if explanation:
            print("   ✅ Grok explanation generated successfully!")
            print(f"   📝 Explanation length: {len(explanation)} characters")
            print(f"   📄 Preview: {explanation[:100]}...")
        else:
            print("   ⚠️  No explanation generated (check logs for details)")
    except Exception as e:
        print(f"   ❌ Failed to generate explanation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Telegram message
    print("\n5️⃣  Testing Telegram message sending...")
    try:
        client.send_success(mock_result, date.today())
        print("   ✅ Telegram message sent successfully!")
        print("   📱 Check your Telegram for the message")
    except Exception as e:
        print(f"   ❌ Failed to send Telegram message: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

