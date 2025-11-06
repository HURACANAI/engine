"""
Integration Examples for Enhanced Features

This module provides integration examples for all enhanced features.
"""

# Example 1: Enhanced Dashboard Integration
def example_enhanced_dashboard():
    """Example: Run enhanced dashboard"""
    import asyncio
    from observability.ui.enhanced_dashboard import EnhancedLiveDashboard
    
    async def run():
        dashboard = EnhancedLiveDashboard(refresh_rate=1.0, capital_gbp=1000.0)
        await dashboard.run()
    
    asyncio.run(run())


# Example 2: Enhanced Daily Report Integration
def example_enhanced_daily_report():
    """Example: Generate enhanced daily report"""
    from observability.analytics.enhanced_daily_report import EnhancedDailyReportGenerator
    from datetime import datetime
    
    generator = EnhancedDailyReportGenerator()
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report = generator.generate_report(date)
    
    formatted = generator.format_report(report)
    print(formatted)
    
    # Save to file
    output_path = f"observability/data/reports/daily_report_{date}.md"
    with open(output_path, 'w') as f:
        f.write(formatted)
    
    print(f"\n✓ Report saved to: {output_path}")


# Example 3: Circuit Breaker Integration
def example_circuit_breaker():
    """Example: Use circuit breaker in trading logic"""
    from src.cloud.training.risk.enhanced_circuit_breaker import EnhancedCircuitBreaker
    
    # Initialize
    circuit_breaker = EnhancedCircuitBreaker(capital_gbp=1000.0)
    
    # Before each trade
    can_trade, reason = circuit_breaker.can_trade(trade_size_gbp=100.0)
    if not can_trade:
        print(f"Trading paused: {reason}")
        return
    
    # Execute trade (simulated)
    pnl_gbp = -5.0  # Loss of £5
    
    # Record outcome
    circuit_breaker.record_trade(pnl_gbp=pnl_gbp, trade_size_gbp=100.0)
    
    # Get status
    status = circuit_breaker.get_status()
    print(f"Level 1: £{status.level1_current:.2f} / £{status.level1_limit:.2f}")
    print(f"Level 2: £{status.level2_current:.2f} / £{status.level2_limit:.2f}")
    print(f"Level 3: £{status.level3_current:.2f} / £{status.level3_limit:.2f}")
    print(f"Level 4: £{status.level4_current:.2f} / £{status.level4_peak:.2f}")


# Example 4: Concept Drift Detection Integration
def example_concept_drift():
    """Example: Use concept drift detection"""
    from src.cloud.training.validation.concept_drift_detector import ConceptDriftDetector, DriftSeverity
    
    # Initialize
    detector = ConceptDriftDetector(
        degradation_threshold=0.10,  # 10% degradation triggers alert
        min_samples=30,
    )
    
    # Record predictions (simulated)
    for i in range(50):
        actual_win = i % 3 != 0  # ~67% win rate
        predicted_prob = 0.75 if actual_win else 0.45
        
        detector.record_prediction(
            actual_win=actual_win,
            predicted_prob=predicted_prob,
            features={'volatility_1h': 1.5, 'momentum': 0.7},
        )
    
    # Check for drift
    drift_report = detector.check_drift()
    
    if drift_report.drift_detected:
        print(f"⚠️ Concept drift detected!")
        print(f"Severity: {drift_report.severity.value}")
        print(f"Degradation: {drift_report.degradation_pct:.1%}")
        print(f"Baseline accuracy: {drift_report.baseline_accuracy:.1%}")
        print(f"Current accuracy: {drift_report.current_accuracy:.1%}")
        
        print("\nRecommendations:")
        for rec in drift_report.recommendations:
            print(f"  • {rec}")
        
        # Trigger retraining if severe
        if drift_report.severity in [DriftSeverity.SEVERE, DriftSeverity.CRITICAL]:
            print("\n🚨 Triggering immediate retraining...")
            # trigger_retraining()
    else:
        print("✅ No concept drift detected")


# Example 5: Confidence-Based Position Scaling Integration
def example_position_scaling():
    """Example: Use confidence-based position scaling"""
    from src.cloud.training.risk.confidence_position_scaler import ConfidenceBasedPositionScaler
    
    # Initialize
    scaler = ConfidenceBasedPositionScaler(
        base_size_gbp=100.0,
        capital_gbp=1000.0,
    )
    
    # Test different confidence levels
    test_cases = [
        {'confidence': 0.85, 'regime': 'TREND', 'regime_confidence': 0.90},
        {'confidence': 0.70, 'regime': 'RANGE', 'regime_confidence': 0.65},
        {'confidence': 0.55, 'regime': 'PANIC', 'regime_confidence': 0.40},
        {'confidence': 0.45, 'regime': 'TREND', 'regime_confidence': 0.50},
    ]
    
    for case in test_cases:
        result = scaler.scale_position(
            confidence=case['confidence'],
            regime=case['regime'],
            regime_confidence=case['regime_confidence'],
            recent_performance=0.75,
            circuit_breaker_active=False,
        )
        
        if result:
            print(f"Confidence {case['confidence']:.2f}: £{result.original_size:.2f} → £{result.scaled_size:.2f} ({result.scale_factor}x)")
            print(f"  Reason: {result.reason}")
        else:
            print(f"Confidence {case['confidence']:.2f}: Trade skipped (confidence too low)")


# Example 6: Explainable AI Integration
def example_explainable_ai():
    """Example: Use explainable AI for decision explanations"""
    from src.cloud.training.models.explainable_ai import ExplainableAIDecisionExplainer
    
    # Initialize
    explainer = ExplainableAIDecisionExplainer()
    
    # Explain a decision
    explanation = explainer.explain_decision(
        features={
            'volatility_1h': 1.5,
            'momentum': 0.7,
            'rsi': 65.0,
            'trend_strength': 0.8,
        },
        predicted_prob=0.75,
        decision='PASS',
        gate_outputs={
            'meta_label': {'value': 0.78, 'passed': True},
            'cost_gate': {'value': 0.85, 'passed': True},
            'confidence_gate': {'value': 0.75, 'passed': True},
        },
        gate_thresholds={
            'meta_label': 0.45,
            'cost_gate': 0.50,
            'confidence_gate': 0.60,
        },
        model_weights={
            'volatility_1h': 0.25,
            'momentum': 0.20,
            'rsi': 0.15,
            'trend_strength': 0.20,
        },
    )
    
    # Display explanation
    print(explainer.format_explanation(explanation))
    
    # Or access individual components
    print("\n" + "=" * 80)
    print("Summary:")
    print(explanation.summary)
    
    print("\nTop Contributors:")
    for contrib in explanation.top_contributors[:5]:
        print(f"  • {contrib.feature_name}: {contrib.value:.3f} ({contrib.direction}, {contrib.importance:.3f})")
    
    if explanation.counterfactuals:
        print("\nCounterfactuals:")
        for cf in explanation.counterfactuals:
            print(f"  • {cf.reason}")


# Example 7: Complete Integration
def example_complete_integration():
    """Example: Complete integration of all features"""
    from src.cloud.training.risk.enhanced_circuit_breaker import EnhancedCircuitBreaker
    from src.cloud.training.risk.confidence_position_scaler import ConfidenceBasedPositionScaler
    from src.cloud.training.validation.concept_drift_detector import ConceptDriftDetector
    from src.cloud.training.models.explainable_ai import ExplainableAIDecisionExplainer
    
    # Initialize all systems
    circuit_breaker = EnhancedCircuitBreaker(capital_gbp=1000.0)
    position_scaler = ConfidenceBasedPositionScaler(base_size_gbp=100.0, capital_gbp=1000.0)
    drift_detector = ConceptDriftDetector()
    explainer = ExplainableAIDecisionExplainer()
    
    # Simulate trading loop
    for i in range(10):
        # 1. Check circuit breaker
        can_trade, reason = circuit_breaker.can_trade()
        if not can_trade:
            print(f"Trade {i+1}: Paused - {reason}")
            continue
        
        # 2. Get signal (simulated)
        confidence = 0.75
        regime = 'TREND'
        features = {'volatility_1h': 1.5, 'momentum': 0.7}
        
        # 3. Scale position based on confidence
        scaling_result = position_scaler.scale_position(
            confidence=confidence,
            regime=regime,
            regime_confidence=0.80,
            recent_performance=0.75,
            circuit_breaker_active=circuit_breaker.trading_paused,
        )
        
        if not scaling_result:
            print(f"Trade {i+1}: Skipped - confidence too low")
            continue
        
        # 4. Execute trade (simulated)
        position_size = scaling_result.scaled_size
        actual_win = i % 3 != 0  # ~67% win rate
        pnl_gbp = 10.0 if actual_win else -5.0
        
        # 5. Record outcome
        circuit_breaker.record_trade(pnl_gbp=pnl_gbp, trade_size_gbp=position_size)
        drift_detector.record_prediction(
            actual_win=actual_win,
            predicted_prob=confidence,
            features=features,
        )
        
        # 6. Explain decision
        explanation = explainer.explain_decision(
            features=features,
            predicted_prob=confidence,
            decision='PASS',
        )
        
        print(f"Trade {i+1}: {'WIN' if actual_win else 'LOSS'} - £{pnl_gbp:.2f} - Size: £{position_size:.2f}")
        print(f"  Top feature: {explanation.top_contributors[0].feature_name if explanation.top_contributors else 'N/A'}")
        
        # 7. Check for drift every 5 trades
        if (i + 1) % 5 == 0:
            drift_report = drift_detector.check_drift()
            if drift_report.drift_detected:
                print(f"  ⚠️ Drift detected: {drift_report.severity.value}")


if __name__ == '__main__':
    print("Enhanced Features Integration Examples")
    print("=" * 80)
    print("\n1. Enhanced Dashboard")
    print("2. Enhanced Daily Report")
    print("3. Circuit Breaker")
    print("4. Concept Drift Detection")
    print("5. Position Scaling")
    print("6. Explainable AI")
    print("7. Complete Integration")
    print("\nRun individual examples or complete integration...")

