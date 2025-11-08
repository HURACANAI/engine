# Monitoring SLOs & Daily Checklist

## Service Level Objectives (SLOs)

### Model Training Pipeline
- **Training Completion Rate**: ≥ 95% of training runs complete successfully
- **Gate Pass Rate**: ≥ 70% of trained models pass all gates
- **Training Duration**: ≤ 2 hours for full curriculum training

### Model Quality
- **Sharpe Ratio**: ≥ 1.0 (target: ≥ 1.5)
- **Win Rate**: ≥ 50% (target: ≥ 55%)
- **Max Drawdown**: ≤ 15%
- **Calibration ECE**: ≤ 0.10
- **Brier Score**: ≤ 0.25

### Data Quality
- **Drift Detection**: No critical drift (PSI ≤ 0.2)
- **Data Completeness**: ≥ 99% complete candles
- **Leakage Detection**: Zero leakage issues

### Production Performance
- **Live/Backtest Consistency**: ≥ 70% (live Sharpe / backtest Sharpe)
- **Execution Slippage**: ≤ 5 bps average
- **System Uptime**: ≥ 99.5%

### Feedback Loop
- **Feedback Latency**: ≤ 5 minutes from trade → Engine
- **Feedback Completeness**: ≥ 95% of trades captured

---

## Daily Checklist

### Morning (09:00 UTC)

- [ ] **Check Active Models**
  ```bash
  python tools/operator_cli.py status
  ```
  - Verify all symbols have active models
  - Check last deployment times
  - Review any overnight alerts

- [ ] **Review Gate Evaluations**
  ```bash
  python tools/operator_cli.py report --days 1
  ```
  - Check yesterday's gate pass rate
  - Review failed gate reasons
  - Identify any concerning trends

- [ ] **Check Data Quality**
  - Review drift reports
  - Verify data completeness
  - Check for any data gaps

- [ ] **Live Performance Review**
  - Compare live vs backtest Sharpe
  - Check consistency ratios
  - Review slippage statistics

### Mid-Day (14:00 UTC)

- [ ] **Training Progress**
  - Check any ongoing training runs
  - Review curriculum progression
  - Monitor calibration improvements

- [ ] **Feedback Loop Health**
  - Verify feedback collector is running
  - Check feedback latency
  - Review any execution anomalies

### Evening (20:00 UTC)

- [ ] **End-of-Day Review**
  - Generate comprehensive report
  ```bash
  python tools/operator_cli.py report --days 1
  ```
  - Review total trades executed
  - Check P&L vs expectations
  - Document any issues

- [ ] **Model Performance**
  - Check Sharpe ratios (live vs backtest)
  - Review opportunity costs (counterfactual)
  - Identify underperforming models

- [ ] **System Health**
  - Review logs for errors/warnings
  - Check disk space
  - Verify database backups

### Weekly (Monday 09:00 UTC)

- [ ] **Weekly Performance Review**
  ```bash
  python tools/operator_cli.py report --days 7
  ```
  - Comprehensive 7-day review
  - Regime performance analysis
  - Calibration drift check

- [ ] **Model Lineage Review**
  - Check model evolution
  - Review rollback history
  - Identify improvement trends

- [ ] **Risk Review**
  - Review max drawdowns by symbol
  - Check position sizing accuracy
  - Verify risk limits are appropriate

---

## Alert Thresholds

### Critical (Immediate Action)
- Gate pass rate < 50%
- Critical drift detected
- Leakage detected
- Live Sharpe < 0.0 (losing money)
- Max drawdown > 15%
- System downtime

### Warning (Review Within 4 Hours)
- Gate pass rate < 70%
- Moderate drift (PSI > 0.15)
- Live/Backtest consistency < 60%
- Slippage > 5 bps average
- Calibration ECE > 0.15

### Info (Review Daily)
- Training completion rate < 95%
- Gate pass rate < 80%
- Live/Backtest consistency < 80%
- Minor drift (PSI > 0.10)

---

## Automated Reports

### Daily Report (Email/Slack at 21:00 UTC)
```
HURACAN ENGINE DAILY REPORT
===========================

Training Activity:
- Sessions: 5
- Models trained: 5
- Gate pass rate: 80% (4/5)

Active Models:
- BTC: btc_v48 (Sharpe: 1.5)
- ETH: eth_v35 (Sharpe: 1.3)
- SOL: sol_v22 (Sharpe: 1.7)

Performance:
- Total trades: 127
- Avg live Sharpe: 1.4
- Consistency: 85%

Quality:
- Drift checks: 3 (all passed)
- Calibration: Good (avg ECE: 0.08)
```

### Weekly Report (Email/Slack Monday 09:00 UTC)
- Comprehensive 7-day analysis
- Model evolution summary
- Counterfactual analysis
- Top learnings/improvements

---

## Monitoring Dashboards

### Real-Time Dashboard
- Active model status
- Live trade P&L
- Gate evaluation results
- Data quality metrics

### Historical Dashboard
- Training success rates over time
- Gate pass rates by gate type
- Model performance trends
- Calibration quality evolution

---

## Incident Response

### Model Performance Degradation
1. Check live consistency ratio
2. Review recent market regime changes
3. Consider rollback if Sharpe < 0.5
4. Investigate via counterfactual analysis

### Gate Failure Spike
1. Review failed gate reasons
2. Check for data quality issues
3. Verify gate thresholds are appropriate
4. Review recent code changes

### Data Quality Issues
1. Check data pipeline
2. Verify data sources
3. Review drift reports
4. Consider training freeze until resolved

### System Downtime
1. Check logs for errors
2. Verify database connectivity
3. Check Hamilton integration
4. Escalate if > 30 minutes

---

## Contact Information

**On-Call Rotation**
- Primary: [Operator contact]
- Secondary: [Backup contact]
- Escalation: [Tech lead contact]

**Monitoring Tools**
- Logs: `observability/logs/`
- Metrics: Enhanced Learning Tracker
- Alerts: [Alert system]

---

Generated: 2025-11-08
