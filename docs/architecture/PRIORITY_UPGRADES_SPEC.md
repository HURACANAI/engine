# Priority Upgrades - Complete Specification

**Date:** 2025-11-11  
**Status:** Implementation In Progress

---

## 🎯 Overview

This document specifies all 10 priority upgrades in order of impact, plus additional requirements for production readiness.

---

## 1. ✅ Consensus Math - Formal Specification

### Formula

**Inputs per engine i:**
- `s_i`: Side in {-1, 0, +1} (sell, wait, buy)
- `p_i`: Probability in [0, 1] (normalized)
- `r_i`: Reliability in [0, 1] (1 - Brier_score with decay)

**Reliability Calculation:**
```
r_i = (1 - Brier_i) * exp(-ln(2) * days_ago / half_life)
```

**Weight Calculation:**
- Base weight: `w_base_i = r_i / sum(r_j)` (normalized reliability)
- Correlation penalty: Minimize `w^T C w` where C is correlation matrix
- Final weight: `w_i = w_base_i * w_corr_i` (normalized)

**Consensus Score:**
```
S = sum_i(w_i * (2*p_i - 1) * s_i)
```

**Adaptive Threshold:**
```
Trade if: |S| > k * sigma_S
Where:
- k = 0.75 (tunable)
- sigma_S = rolling_std(S, window=100)
```

**Implementation:** ✅ `src/cloud/training/consensus/formal_consensus.py`

---

## 2. ✅ Regime Gating - Hard Gates

### Specification

**Regime Classification:**
- TREND: ADX > 25, strong directional move
- RANGE: ADX < 20, moderate volatility (1-4%)
- PANIC: Volatility > 5%, negative trend, high volume
- ILLIQUID: Volume < 30% or spread > 50 bps

**Engine Approval Matrix:**
- Each engine has approved_regimes set
- Only approved engines vote in each regime
- Leaderboard refreshed weekly per regime

**Implementation:** ✅ `src/cloud/training/regime/hard_gates.py`

---

## 3. ✅ Purged Walk-Forward Testing

### Specification

**Window Structure:**
- Train window: 120 days
- Purge gap: 3-5 days (removes label overlap)
- Test window: 30 days
- Window type: Expanding or rolling

**Combinatorial Purged K-Fold:**
- K folds for model selection
- Purge gap between train and validation
- Reduces data snooping

**Reporting:**
- Mean and dispersion (std, min, max, median) for all metrics
- Stability across years and coins

**Implementation:** ✅ `src/cloud/training/validation/purged_walk_forward.py`

---

## 4. ✅ Full Cost Model

### Specification

**Cost Components:**
1. **Slippage:**
   ```
   Slippage = spread_cost + market_impact
   spread_cost = spread_bps / 2
   market_impact = sqrt(participation_rate) * 5 bps
   ```

2. **Fees:**
   - Maker fee: 2-5 bps (venue-dependent)
   - Taker fee: 4-5 bps (venue-dependent)

3. **Funding Costs:**
   ```
   Funding = funding_rate_bps * (hours_held / 8)
   ```

4. **Latency Penalty:**
   ```
   Penalty = (latency_ms / 10) * 0.1 bps
   ```

**Safety Margin:**
- Reject trades where: `expected_edge - total_costs < safety_margin_bps`
- Default safety margin: 3.0 bps

**Implementation:** ✅ `src/cloud/training/costs/full_cost_model.py`

---

## 5. ✅ Position Sizing - Volatility Targeting

### Specification

**Base Size Calculation:**
```
target_risk_usd = equity * target_risk_pct
base_size = target_risk_usd / (price * volatility_risk)
```

**Kelly Fraction:**
```
Kelly = (p * b - q) / b
Where:
- p = win_rate
- q = 1 - p
- b = avg_win / avg_loss

Final size = base_size * (Kelly / 4)  # Conservative cap
```

**Caps:**
- Single trade risk: 0.5-1.5% of equity
- Single coin exposure: 25% of equity
- Max leverage: 3x (configurable)
- Total open risk: Sum of stop losses ≤ daily loss stop

**Implementation:** ✅ `src/cloud/training/risk/volatility_targeting.py`

---

## 6. ✅ Circuit Breakers

### Specification

**Daily Drawdown Breaker:**
- Hard stop: 3% of equity (configurable)
- Action: Stop trading for the day
- Reset: Next day at 00:00

**Streak Breaker:**
- Trigger: 3 consecutive losses
- Action: Halve position sizes
- Reset: On next win

**Volatility Breaker:**
- Trigger: Realized vol > 95th percentile
- Action: Switch to defense mode (reduce size or flat)
- Reset: When volatility drops

**Implementation:** ✅ `src/cloud/training/risk/circuit_breakers.py`

---

## 7. ✅ Portfolio Optimization Layer

### Specification

**Optimization Objective:**
```
Maximize: sum(w_i * expected_edge_i * confidence_i) - penalties
Subject to:
- sum(w_i) = 1
- 0 <= w_i <= 1
- HHI <= max_hhi
- Total exposure <= risk_budget
```

**Penalties:**
- Turnover penalty: `0.1 * turnover`
- Concentration penalty: `max(0, HHI - max_hhi) * 100`

**Constraints:**
- Risk budget: 70% of equity
- Cash buffer: 10% of equity
- Max HHI: 0.25

**Implementation:** ✅ `src/cloud/training/portfolio/optimizer.py`

---

## 8. ✅ Execution Engine - Smart Router

### Specification

**Venue Scoring:**
```
Score = 0.3*spread_score + 0.2*queue_score + 0.2*latency_score + 
        0.15*reliability_score + 0.15*size_score
```

**Execution Strategies:**
- **Post-only maker:** When spread > 10 bps
- **TWAP:** For persistent signals (>70% persistence)
- **POV:** For large orders (>10% of avg volume)
- **Iceberg:** For very large orders (>$50k)
- **Limit/Market:** For small orders

**Implementation:** ✅ `src/cloud/training/execution/smart_router.py`

---

## 9. 📋 Live-Shadow and Canary Deployment

### Specification

**Shadow Trading:**
- Run new models in shadow for 2-4 weeks
- Compare metrics: PnL, hit rate, Sharpe, IR, drawdown, turnover, cost per trade
- Statistical significance test (t-test or bootstrap)

**Promotion Criteria:**
- New model beats baseline with p < 0.05
- All metrics improved or equal
- No degradation in any critical metric

**Implementation:** 🚧 Planned

---

## 10. 📋 Observability - Standard Events

### Specification

**Event Schema:**
```python
{
    "timestamp": datetime,
    "symbol": str,
    "features_hash": str,  # Hash of features used
    "engine_votes": Dict[str, Dict],  # engine_id -> {side, prob, reliability}
    "consensus_score": float,
    "action": str,  # "buy", "sell", "wait"
    "size_usd": float,
    "price": float,
    "expected_edge_bps": float,
    "realized_pnl": Optional[float],  # Filled after trade closes
    "regime": str,
    "costs_bps": float,
}
```

**Health Checks:**
- Data gaps
- Model load times
- Latency
- Error rates

**Dashboards:**
- PnL by engine
- PnL by regime
- PnL by venue
- Cost breakdown

**Implementation:** 🚧 Planned

---

## 📋 Additional Requirements

### Risk Engine Presets

**Conservative:**
- Per trade risk: 0.25%
- Daily loss stop: 1.5%
- Max leverage: 1.5x

**Balanced:**
- Per trade risk: 0.5%
- Daily loss stop: 2.5%
- Max leverage: 2x

**Aggressive:**
- Per trade risk: 1.0%
- Daily loss stop: 3.5%
- Max leverage: 3x

**All presets enforce:**
- Single coin cap: 25% of equity
- Sector cap: 50% of equity
- Total open risk: Sum of stops ≤ daily loss stop

### Backtest Reports

**Required Metrics:**
- Net and gross Sharpe, Sortino, Calmar
- Hit rate, avg win, avg loss, profit factor
- Max drawdown, avg drawdown, time to recover
- Turnover, trades per day, avg holding time
- Slippage share of gross alpha
- Fee share, funding share
- Capacity estimate
- Stability (dispersion across years and coins)

### Data Governance

**Rules:**
- Vendor reconciliation: Cross-check 2+ sources
- Missing data: Forward fill only small gaps (< 1 hour), drop large gaps
- Time alignment: Snap to exchange timestamps
- Outlier filter: Robust z-score on returns, clip tails
- Survivorship bias: Include delisted coins in tests
- Universe selection: Liquidity and age filters, defined upfront

### Security and Safety

**Requirements:**
- Secrets in vault (rotated)
- IP allowlist per venue
- Read-only keys for research
- Trade keys only on live box
- Dry run mode default in new environments
- Kill switch: Set size to zero, cancel all orders

### Daily Schedule

**02:00 UTC:**
- Data ingest
- Integrity checks
- Feature build
- Model training
- Walk-forward testing
- Champion vs challenger selection
- Publish models

**Every 5 minutes:**
- Regime detection
- Signal build
- Consensus compute
- Risk check
- Route orders if threshold cleared

**Every 15 minutes:**
- Health check
- Latency report

**Hourly:**
- Funding and basis scan
- Adjust positions if carry turns

**23:55 UTC:**
- PnL attribution (by engine, by cost)
- Snapshot to reports

---

## ✅ Implementation Status

| Upgrade | Status | File |
|---------|--------|------|
| 1. Consensus Math | ✅ Complete | `consensus/formal_consensus.py` |
| 2. Regime Gating | ✅ Complete | `regime/hard_gates.py` |
| 3. Purged Walk-Forward | ✅ Complete | `validation/purged_walk_forward.py` |
| 4. Full Cost Model | ✅ Complete | `costs/full_cost_model.py` |
| 5. Position Sizing | ✅ Complete | `risk/volatility_targeting.py` |
| 6. Circuit Breakers | ✅ Complete | `risk/circuit_breakers.py` |
| 7. Portfolio Optimization | ✅ Complete | `portfolio/optimizer.py` |
| 8. Execution Router | ✅ Complete | `execution/smart_router.py` |
| 9. Canary Deployment | 🚧 Planned | TBD |
| 10. Observability | 🚧 Planned | TBD |

---

**See individual module files for detailed implementation.**

