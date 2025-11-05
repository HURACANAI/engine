# Critical Gaps Between Documentation and Implementation

**Status**: 10 gaps identified, prioritized for implementation
**Date**: 2025-11-05

---

## Overview

The documentation (v5.0) describes a complete system, but several critical components are either:
1. **Missing entirely** (not implemented)
2. **Partially implemented** (logic exists but not wired)
3. **Inconsistent** (doc says X, code does Y)

---

## Gap 1: Action Space Mismatch ⚠️ CRITICAL

**Issue**: Documentation lists 8 RL actions but gate systems reference actions not in the list.

**What Doc Says**:
```python
Actions: HOLD, ENTER_SMALL, ENTER_MEDIUM, ENTER_LARGE,
         ADD_TO_WINNER, ADD_TO_LOSER, EXIT_PARTIAL, EXIT_ALL
```

**What Gates Reference**:
- SCRATCH (immediate exit)
- UNLOCK_RUNNER (keep/exit runner)
- TRAIL_RUNNER (adjust trailing stop)

**Fix Required**:
Either:
1. Add these to RL action space (expand to 11 actions)
2. Make them deterministic rules outside RL (recommended)

**Recommendation**: Make deterministic rules
```python
# AFTER RL policy decides ENTER_MEDIUM:
if scratch_policy.should_scratch(entry_result):
    execute_action(SCRATCH)  # Override RL decision

# AFTER TP1 hit:
if runner_unlocker.should_unlock_runner(evidence):
    keep_runner()  # Deterministic rule
else:
    exit_all()
```

**Status**: ⚠️ Needs clarification in docs

---

## Gap 2: Dual-Mode Books Missing ⚠️ HIGH PRIORITY

**Issue**: Doc describes "scalp vs runner" logic, but no formal dual-book architecture.

**What's Missing**:
- `book_short`: Fast scalps (£1-£2 target, 5-15 sec hold)
- `book_long`: Runners (£5-£20 target, 5-60 min hold)
- Per-asset profiles (ETH/SOL/BTC = BOTH allowed)
- Independent caps and exit ladders per book

**Current State**: Single position manager

**Fix Required**:
```python
@dataclass
class DualBookManager:
    """Manage separate short-hold and long-hold books."""

    book_short: Dict[str, Position]  # Scalps
    book_long: Dict[str, Position]   # Runners

    # Per-asset profiles
    asset_profiles: Dict[str, AssetProfile]

    # Independent caps
    max_short_heat: float = 0.40  # Max 40% in scalps
    max_long_heat: float = 0.50   # Max 50% in runners
```

**Status**: 🔴 NOT IMPLEMENTED

---

## Gap 3: Purged Walk-Forward CV Missing ⚠️ CRITICAL

**Issue**: Triple-barrier labels exist, but no purged cross-validation to prevent leakage.

**What's Missing**:
- Combinatorial Purged K-Fold
- Embargo periods
- OOS pass requirement before deployment

**Current State**: Standard K-fold (leaks information!)

**Fix Required**:
```python
from mlfinlab.cross_validation import CombinatorialPurgedKFold

cv = CombinatorialPurgedKFold(
    n_splits=5,
    n_test_splits=2,
    embargo_pct=0.01,  # 1% embargo
)

for train_idx, test_idx in cv.split(X, pred_times=trade_times):
    # Train on purged folds
    model.fit(X[train_idx], y[train_idx])

    # Test
    preds = model.predict(X[test_idx])
    oos_performance.append(score(y[test_idx], preds))

# BLOCK deployment if OOS < threshold
if np.mean(oos_performance) < production_threshold:
    raise ValueError("OOS performance insufficient!")
```

**Status**: 🔴 NOT IMPLEMENTED

---

## Gap 4: Cost Model Incompleteness ⚠️ HIGH

**Issue**: Cost gate calculates costs, but maker rebates and post-only preference not explicit.

**What's Missing**:
- Maker rebate accounting (-2 bps)
- Post-only flag for LIMIT orders
- Explicit maker-bias config

**Current Implementation**: Charges maker fee as cost (wrong!)

**Fix Required**:
```python
@dataclass
class CostEstimate:
    exchange_fee_bps: float  # Can be NEGATIVE (rebate)

def _estimate_exchange_fee(self, order_type):
    if order_type == OrderType.MAKER:
        return -self.maker_rebate_bps  # NEGATIVE = we get paid
    else:
        return self.taker_fee_bps

# In smart executor
if self.maker_bias and fill_prob > 0.60:
    use_post_only_limit()  # Prefer maker
elif eps_positive_even_with_taker:
    use_taker()  # Fallback
else:
    skip_trade()  # Would lose money
```

**Status**: 🟡 PARTIAL (needs rebate logic)

---

## Gap 5: Fill Probability Loop Missing ⚠️ MEDIUM

**Issue**: Fill probability calculator exists, but no feedback loop measuring accuracy.

**What's Missing**:
- Actual fill time tracking
- Model error calculation
- Maker ratio dashboard
- Stale order detection

**Fix Required**:
```python
@dataclass
class FillMetrics:
    predicted_time_sec: float
    actual_time_sec: float
    prediction_error: float  # actual - predicted
    was_filled: bool
    order_type: OrderType

# Dashboard KPIs
kpis = {
    'maker_ratio': filled_maker / total_fills,
    'avg_fill_time_error': np.mean(errors),
    'stale_order_rate': cancelled / total,
    'fill_accuracy_1s': hits_within_1s / predictions,
}

# Alert if degraded
if kpis['fill_time_error'] > 5.0:  # >5 sec error
    alert("Fill probability model degraded")
```

**Status**: 🟡 PARTIAL (calculator exists, feedback missing)

---

## Gap 6: Gate Counterfactuals Missing ⚠️ HIGH

**Issue**: Gates block trades, but we don't track "what would have happened."

**What's Missing**:
- Counterfactual P&L tracking
- Per-gate value attribution
- Auto-tuning thresholds based on counterfactuals

**Fix Required**:
```python
@dataclass
class GateDecision:
    gate_name: str
    blocked: bool
    reason: str
    features: Dict

    # NEW: Counterfactual tracking
    predicted_pnl: Optional[float] = None  # What model predicted
    actual_pnl: Optional[float] = None     # What actually happened

# After N minutes, check if blocked trade would have won
if decision.blocked:
    simulated_result = simulate_trade(
        entry_price=current_price,
        exit_price=price_N_minutes_later,
        direction=direction,
    )

    decision.actual_pnl = simulated_result.pnl

    # Update gate statistics
    gate.record_counterfactual(
        blocked=True,
        would_have_won=simulated_result.pnl > 0,
    )

# Auto-tune threshold
if gate.false_negative_rate > 0.20:  # Blocking too many winners
    gate.threshold *= 0.95  # Loosen
elif gate.false_positive_rate > 0.30:  # Letting through losers
    gate.threshold *= 1.05  # Tighten
```

**Status**: 🔴 NOT IMPLEMENTED

---

## Gap 7: Regime Transition → Weight Tilting Missing ⚠️ MEDIUM

**Issue**: Regime anticipator predicts transitions, but doesn't pre-adjust engine weights.

**What's Missing**:
- Pre-tilting weights before transition
- Gradual weight shifting
- Clip size reduction during transition

**Fix Required**:
```python
# When transition detected
if transition.predicted == "TREND → PANIC":
    confidence = transition.confidence  # 0.85

    # Pre-tilt weights (before flip happens)
    engine_weights['trend'] *= (1 - confidence * 0.5)  # Down-weight trend
    engine_weights['range'] *= (1 - confidence * 0.3)  # Down-weight range
    # (PANIC engines stay same or up)

    # Reduce clip sizes
    for position in open_positions:
        position.max_add_size *= (1 - confidence * 0.5)

# After transition completes
if regime_actually_flipped:
    # Full weight update
    update_weights_for_new_regime()
```

**Status**: 🟡 PARTIAL (anticipator exists, tilting missing)

---

## Gap 8: Engine Health Freeze Hysteresis Missing ⚠️ MEDIUM

**Issue**: Engine health monitor freezes engines, but no hysteresis to prevent thrashing.

**What's Missing**:
- Freeze/unfreeze thresholds (different)
- Minimum trades before unfreeze
- Gradual re-weighting

**Fix Required**:
```python
@dataclass
class EngineHealthConfig:
    freeze_threshold: float = 0.30    # Freeze if health < 0.30
    unfreeze_threshold: float = 0.45  # Unfreeze if health > 0.45
    min_trades_frozen: int = 20       # Stay frozen for 20 trades min

class EngineHealthMonitor:
    def update_engine_status(self, engine_name, health):
        current_status = self.status[engine_name]

        if current_status == Status.ACTIVE:
            if health.penalty < self.config.freeze_threshold:
                self.freeze_engine(engine_name)

        elif current_status == Status.FROZEN:
            trades_frozen = self.trades_since_freeze[engine_name]

            # Need BOTH: enough trades AND health improved
            if (trades_frozen >= self.config.min_trades_frozen and
                health.penalty > self.config.unfreeze_threshold):
                self.unfreeze_engine(engine_name)
```

**Status**: 🟡 PARTIAL (monitor exists, hysteresis missing)

---

## Gap 9: Macro Cooldown + Re-Warm Missing ⚠️ MEDIUM

**Issue**: Macro detector pauses trading, but no gradual re-entry schedule.

**What's Missing**:
- Cooldown enforcement
- Staged size ramp after event
- Re-warm schedule

**Fix Required**:
```python
@dataclass
class MacroCooldown:
    paused: bool = False
    pause_until: float = 0.0
    re_warm_phase: int = 0  # 0=normal, 1-3=warming

    # Re-warm schedule
    re_warm_size_multipliers = [0.25, 0.50, 0.75, 1.0]
    re_warm_duration_minutes = 15

def handle_macro_event(detection):
    if detection.severity == EventSeverity.EXTREME:
        # Pause
        cooldown.paused = True
        cooldown.pause_until = time.time() + (detection.pause_duration_minutes * 60)
        cooldown.re_warm_phase = 1

        logger.critical("PAUSE_TRADING", duration=detection.pause_duration_minutes)

        # Exit all if recommended
        if detection.recommended_action == TradingAction.EXIT_ALL:
            exit_all_positions()

def check_cooldown():
    if cooldown.paused and time.time() >= cooldown.pause_until:
        # Start re-warming
        cooldown.paused = False
        cooldown.re_warm_phase = 1
        logger.info("Re-warming phase 1 (25% size)")

    # Advance re-warm phases
    if cooldown.re_warm_phase > 0:
        time_in_phase = time.time() - cooldown.phase_start_time
        if time_in_phase > (cooldown.re_warm_duration_minutes * 60):
            cooldown.re_warm_phase += 1
            if cooldown.re_warm_phase > 3:
                cooldown.re_warm_phase = 0  # Back to normal

# Apply size multiplier
if cooldown.re_warm_phase > 0:
    position_size *= cooldown.re_warm_size_multipliers[cooldown.re_warm_phase]
```

**Status**: 🟡 PARTIAL (detector exists, cooldown missing)

---

## Gap 10: Shadow A/B Promotion Criteria Missing ⚠️ HIGH

**Issue**: Rollout plan exists, but no formal promotion criteria.

**What's Missing**:
- A/B comparison logic
- Statistical significance test
- Minimum outperformance duration
- Automatic promotion gate

**Fix Required**:
```python
@dataclass
class PromotionCriteria:
    min_consecutive_days: int = 5
    min_net_bps_improvement: float = 10.0  # 10 bps/day better
    min_expected_shortfall_95: float = 0.0  # ES(95%) must be positive

def check_promotion(shadow_performance, prod_performance):
    """Check if shadow can be promoted to production."""

    # 1. Consecutive days better
    consecutive_wins = 0
    for day in last_N_days:
        if shadow_performance[day] > prod_performance[day]:
            consecutive_wins += 1
        else:
            consecutive_wins = 0

    if consecutive_wins < criteria.min_consecutive_days:
        return False, f"Only {consecutive_wins} consecutive days"

    # 2. Net bps improvement
    shadow_avg = np.mean(shadow_performance)
    prod_avg = np.mean(prod_performance)
    improvement = shadow_avg - prod_avg

    if improvement < criteria.min_net_bps_improvement:
        return False, f"Only {improvement:.1f} bps improvement"

    # 3. Expected Shortfall at 95% (risk-adjusted)
    shadow_es95 = np.percentile(shadow_performance, 5)  # 5th percentile

    if shadow_es95 < criteria.min_expected_shortfall_95:
        return False, f"ES(95%) = {shadow_es95:.1f} (negative tail)"

    # 4. Statistical significance (t-test)
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(shadow_performance, prod_performance)

    if p_value > 0.05:
        return False, f"Not statistically significant (p={p_value:.3f})"

    # All checks passed
    return True, f"PROMOTE: {improvement:.1f} bps better, ES(95%)={shadow_es95:.1f}, p={p_value:.3f}"

# In rollout script
if check_promotion(shadow_perf, prod_perf)[0]:
    logger.info("PROMOTING shadow to production")
    promote_to_production()
else:
    reason = check_promotion(shadow_perf, prod_perf)[1]
    logger.info(f"NOT promoting: {reason}")
```

**Status**: 🔴 NOT IMPLEMENTED

---

## Additional Monitoring Gaps

**What's Missing from Dashboards**:

1. **Cost Model Error**
   - Modeled cost vs realized cost
   - Alert if error > 3 bps

2. **Gate Economics**
   - Counterfactual P&L per gate
   - Value attribution

3. **Maker Ratio / Fill Time**
   - Target: 60%+ maker
   - Target: <3 sec fill time error

4. **Per-Engine Health Dashboard**
   - WR by regime
   - Health score
   - Freeze status
   - Trades since freeze

5. **Transition Anticipator Accuracy**
   - Predictions vs actual flips
   - Lead time distribution
   - False positive rate

6. **Runner Contribution**
   - P&L from runners vs scalps
   - Giveback after TP1
   - Runner WR vs scalp WR

---

## Implementation Priority

### P0 (CRITICAL - Must Fix Before Production)
1. ✅ Purged CV (Gap 3) - Prevents leakage
2. ✅ Cost Model Rebates (Gap 4) - Affects profitability calculations
3. ✅ Gate Counterfactuals (Gap 6) - Proves gate value
4. ✅ Shadow A/B Criteria (Gap 10) - Safe promotion

### P1 (HIGH - Fix in First Month)
5. ✅ Dual-Mode Books (Gap 2) - Core architecture
6. ✅ Action Space Clarification (Gap 1) - Consistency
7. ✅ Freeze Hysteresis (Gap 8) - Prevents thrashing

### P2 (MEDIUM - Fix in First Quarter)
8. ✅ Fill Probability Loop (Gap 5) - Quality improvement
9. ✅ Regime Transition Tilting (Gap 7) - Performance boost
10. ✅ Macro Cooldown (Gap 9) - Risk management

---

## Next Steps

1. **Review this document** - Confirm priorities
2. **Implement P0 fixes** - Critical gaps first
3. **Update documentation** - Reflect actual state
4. **Add monitoring** - Track all KPIs
5. **Test in shadow mode** - Validate fixes
6. **Gradual rollout** - Use promotion criteria

---

**Status**: 10 gaps identified, prioritized, fixes specified
**Owner**: System architect
**Target**: P0 by Week 2, P1 by Month 1, P2 by Quarter 1
