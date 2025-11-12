#!/usr/bin/env python3
"""Quick training stats viewer"""

import psycopg2
from datetime import datetime

conn = psycopg2.connect(host='localhost', port=5432, database='huracan', user='haq')
cur = conn.cursor()

# Get stats
cur.execute("""
    SELECT
        COUNT(*) as total_trades,
        COUNT(*) FILTER (WHERE is_winner = true) as winning_trades,
        COUNT(*) FILTER (WHERE is_winner = false) as losing_trades,
        COALESCE(SUM(net_profit_gbp), 0) as total_profit,
        COALESCE(AVG(CASE WHEN is_winner THEN net_profit_gbp END), 0) as avg_profit,
        COALESCE(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as avg_loss
    FROM trade_memory
    WHERE symbol = 'SOL/USDT'
""")

row = cur.fetchone()
total, wins, losses, profit, avg_win, avg_loss = row
win_rate = (wins / total * 100) if total > 0 else 0

print('=' * 80)
print('  SOL/USDT TRAINING DASHBOARD')
print('  Updated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('=' * 80)
print()
print(f'  Total Trades:     {total:,}')
print(f'  Winning Trades:   {wins:,}')
print(f'  Losing Trades:    {losses:,}')
print(f'  Win Rate:         {win_rate:.1f}%')
print(f'  Total P&L:        £{profit:.2f}')
print(f'  Avg Win:          £{avg_win:.2f}')
print(f'  Avg Loss:         £{avg_loss:.2f}')
print()

# Get recent trades
cur.execute("""
    SELECT trade_id, entry_price, exit_price, net_profit_gbp, is_winner, model_confidence
    FROM trade_memory
    WHERE symbol = 'SOL/USDT'
    ORDER BY trade_id DESC
    LIMIT 10
""")

print('  Recent Trades (Last 10):')
print('  ' + '-' * 76)
print(f'  {"ID":>6} | {"Entry":>8} | {"Exit":>8} | {"P&L (£)":>10} | {"Result":>8} | {"Conf":>6}')
print('  ' + '-' * 76)

for row in cur.fetchall():
    tid, entry, exit_p, profit_val, is_win, conf = row
    result = 'WIN' if is_win else 'LOSS'
    print(f'  {tid:>6} | ${float(entry):>7.2f} | ${float(exit_p):>7.2f} | {float(profit_val):>+9.2f} | {result:>8} | {float(conf):>5.2f}')

print('=' * 80)
conn.close()
