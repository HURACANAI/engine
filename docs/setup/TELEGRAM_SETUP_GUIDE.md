# 📱 Telegram Monitoring Setup Guide

## Quick Setup

### Step 1: Get Your Chat ID

Run this script to get your Telegram chat ID:

```bash
python scripts/get_telegram_chat_id.py
```

**Instructions:**
1. The script will start and wait for a message
2. Open Telegram and find your bot
3. Send any message to the bot (e.g., `/start` or `Hello`)
4. The script will print your `chat_id`
5. Copy the `chat_id` value

### Step 2: Update Config

Edit `config/base.yaml` and add your chat_id:

```yaml
notifications:
  telegram_enabled: true
  telegram_webhook_url: "https://api.telegram.org/bot8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0/sendMessage"
  telegram_chat_id: "YOUR_CHAT_ID_HERE"  # Paste the chat_id from Step 1
```

### Step 3: Test

Run the engine and check Telegram for notifications:

```bash
python -m src.cloud.training.pipelines.daily_retrain
```

---

## What You'll Receive

### ✅ **System Events**
- 🚀 Engine startup (which coins are being trained)
- 🏁 Engine completion (summary stats)
- ✅ Health checks (system status)
- 🚨 Errors (with context)

### 📊 **Training Progress**
- Every trade executed (entry/exit)
- Every trade closed (P&L)
- Learning updates (what it learned)
- Model updates (when models improve)
- Gate decisions (why trades blocked/allowed)

### 🚨 **Validation Failures**
- OOS validation failures (HARD BLOCK)
- Overfitting detection
- Data quality issues
- Performance degradation

### 📈 **Performance Summaries**
- Hourly summaries (if enabled)
- Daily summaries
- Best/worst performing symbols

---

## Log Files

All notifications are also logged to:
```
logs/engine_monitoring_YYYYMMDD_HHMMSS.log
```

You can copy/paste this file to share with support.

---

## Notification Levels

- **CRITICAL** 🚨: Validation failures, errors, system failures
- **HIGH** ⚠️: Warnings, degraded performance
- **MEDIUM** ℹ️: Regular updates, trades, health checks
- **LOW** 📊: Info only, learning updates

---

## Troubleshooting

### No Messages Received?

1. **Check chat_id**: Make sure it's correct in `config/base.yaml`
2. **Check bot token**: Should be `8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0`
3. **Check Telegram**: Make sure you sent a message to the bot first
4. **Check logs**: Look for `telegram_send_failed` in logs

### Too Many Messages?

Edit `src/cloud/training/pipelines/daily_retrain.py` and change `min_notification_level`:

```python
min_notification_level=NotificationLevel.MEDIUM,  # Only MEDIUM and above
```

### Rate Limiting?

Telegram limits to 20 messages/minute. The system automatically handles this.

---

## Example Messages

### System Startup
```
🚀 *System Startup*

Training on 20 coins
Symbols: BTC/USDT, ETH/USDT, SOL/USDT ... and 17 more
```

### Validation Failure
```
🚨 *Validation Failed*

Type: OOS Validation
Reason: OOS Sharpe (0.85) below minimum (1.0)
Symbol: BTC/USDT

Details:
sharpe: 0.85
min_required: 1.0

⚠️ *Action Required:* Fix before deploying!
```

### Health Check
```
✅ *Health Check*

Status: HEALTHY
Services: 8/8 healthy

🚨 *Alerts (0):*

⚠️ *Warnings (0):*
```

---

## Next Steps

1. ✅ Get your chat_id
2. ✅ Update config
3. ✅ Run the engine
4. ✅ Monitor Telegram
5. ✅ Check log files

You're all set! 🎉

