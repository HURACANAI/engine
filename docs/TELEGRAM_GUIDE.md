# Telegram Monitoring Guide

Complete guide to using Telegram monitoring with your Engine.

## Overview

The Engine sends **automatic notifications** to Telegram about:
- ✅ Training progress (started, downloading, completed, failed)
- ✅ Batch progress updates
- ✅ System health checks
- ✅ Validation failures
- ✅ Errors and warnings
- ✅ Shadow trades (if enabled)
- ✅ Model updates

**Note**: Currently, the Telegram bot is **one-way** (sends notifications to you). It doesn't respond to commands yet.

## Setup

### 1. Get Your Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the bot token (e.g., `8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0`)

### 2. Get Your Chat ID

Run the helper script:
```bash
python scripts/get_telegram_chat_id.py
```

Then:
1. Open Telegram and find your bot
2. Send any message to the bot (e.g., `/start` or `Hello`)
3. The script will print your chat ID

### 3. Configure Environment Variables

Set these environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

Or add to your config file:
```yaml
telegram:
  bot_token: "your_bot_token_here"
  chat_id: "your_chat_id_here"
```

## Available Notifications

### Training Progress Notifications

You'll receive notifications for:
- **Training Started**: When a coin starts training
  ```
  📊 Training Started
  
  Symbol: BTC/USDT
  Batch: 1/10
  Task: 1/2
  ```

- **Downloading Data**: When downloading historical data
  ```
  ⬇️ Downloading Data
  
  Symbol: BTC/USDT
  Window: 150 days
  This may take several minutes...
  ```

- **Data Downloaded**: When data download completes
  ```
  ✅ Data Downloaded
  
  Symbol: BTC/USDT
  Rows: 216,000
  Starting model training...
  ```

- **Training Complete**: When training finishes
  ```
  ✅ Training Complete
  
  Symbol: BTC/USDT
  Status: ✅ Published
  Reason: Model passed all validations
  ```

- **Training Failed**: When training fails
  ```
  ❌ Training Failed
  
  Symbol: BTC/USDT
  Error: Task exceeded 30 minute timeout
  ```

### Batch Progress Notifications

You'll receive updates about batch progress:
```
📦 Batch Progress

Batch: 1/10
Progress: 2/2 tasks (100%)
Symbols: BTC/USDT, ETH/USDT
```

### System Health Notifications

Health check results:
```
✅ System Health Check

Status: HEALTHY
Services: 5/5 healthy
Database: Connected
CPU: 2.1%
Memory: 20.2%
```

### Error Notifications

When errors occur:
```
🚨 Error Detected

Type: AttributeError
Message: 'ComprehensiveTelegramMonitor' object has no attribute 'notifytrainingprogress'
Context: timestamp: 2025-11-06T23:25:18.919653+00:00

⚠️ Action Required
```

### Validation Failure Notifications

When models fail validation:
```
⚠️ Validation Failure

Type: Model Validation
Symbol: BTC/USDT
Reason: Coverage 0.1852 below threshold
Details: {...}
```

## Notification Levels

Notifications are filtered by priority level:

- **CRITICAL**: Immediate alerts (errors, system failures)
- **HIGH**: Important updates (validation failures, health issues)
- **MEDIUM**: Regular updates (trades, training progress)
- **LOW**: Info only (batch progress, minor updates)

You can configure the minimum level in settings.

## Rate Limiting

The bot is rate-limited to **20 messages per minute** to avoid Telegram API limits. If you receive too many notifications, some may be batched or delayed.

## Logging

All notifications are also logged to a file:
- Location: `logs/engine_monitoring_YYYYMMDD_HHMMSS.log`
- Format: JSON with timestamps
- Includes: All notifications, errors, and system events

## Troubleshooting

### Not Receiving Notifications

1. **Check Bot Token**: Make sure `TELEGRAM_BOT_TOKEN` is set correctly
2. **Check Chat ID**: Make sure `TELEGRAM_CHAT_ID` is set correctly
3. **Check Bot**: Send a message to your bot to make sure it's working
4. **Check Logs**: Look for errors in the logs
5. **Check Rate Limits**: You may be hitting Telegram's rate limits

### Too Many Notifications

1. **Increase Minimum Level**: Set `min_notification_level` to `MEDIUM` or `HIGH`
2. **Disable Specific Types**: Disable notifications you don't need:
   ```python
   enable_trade_notifications=False,
   enable_learning_notifications=False,
   ```

### Bot Not Responding

The bot is currently **one-way** (sends notifications only). It doesn't respond to commands yet.

## Future: Interactive Commands

We're planning to add interactive commands like:
- `/status` - Get current system status
- `/progress` - Get training progress
- `/health` - Get health check results
- `/trades` - Get recent shadow trades
- `/engines` - Get engine status
- `/pause` - Pause training
- `/resume` - Resume training

These will be added in a future update.

## Example Notifications

### Training Session Start
```
🚀 Engine Started

Training on 20 coins
Symbols: BTC/USDT, ETH/USDT, SOL/USDT ... and 17 more
```

### Training Session Complete
```
🏁 Engine Completed

Total Trades: 150
Total Profit: £1,234.56
Duration: 45 minutes
```

### Health Check
```
✅ System Health Check

Status: HEALTHY
Services: 5/5 healthy
Database: Connected
CPU: 2.1%
Memory: 20.2%
Disk: 6.5%
```

## Summary

**What You Get:**
- ✅ Real-time training progress
- ✅ Batch progress updates
- ✅ System health checks
- ✅ Error notifications
- ✅ Validation failures
- ✅ All logged to file

**What You Can Do:**
- Monitor training progress
- Get alerts for errors
- Track system health
- Review logs for debugging

**What's Coming:**
- Interactive commands (`/status`, `/progress`, etc.)
- Custom notification filters
- Notification scheduling
- Multi-chat support

