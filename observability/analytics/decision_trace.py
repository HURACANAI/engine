"""
Decision Trace

Tracks the complete decision timeline from signal to execution with timing.

Provides:
- Full trace: Signal → Gates → Execution
- Timing breakdown (which step took longest)
- Bottleneck identification
- Parallel vs sequential execution analysis

This answers: "Where is the system spending time? Which gates are slowest?"

Usage:
    tracer = DecisionTracer()

    # Start trace
    trace_id = tracer.start_trace(signal_id="sig_001", symbol="ETH-USD")

    # Record steps
    tracer.record_step(trace_id, "signal_received", latency_ms=0.1)
    tracer.record_step(trace_id, "meta_label_gate", latency_ms=2.3, result="PASS")
    tracer.record_step(trace_id, "cost_gate", latency_ms=0.8, result="PASS")
    ...

    # Finish trace
    tracer.finish_trace(trace_id, outcome="executed")

    # Analyze
    analysis = tracer.analyze_trace(trace_id)
    print(analysis.timeline)
    print(analysis.bottlenecks)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TraceStep:
    """Single step in decision trace"""
    step_name: str
    start_time_ms: float
    end_time_ms: float
    latency_ms: float
    result: Optional[str] = None  # "PASS", "FAIL", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    """Complete decision trace"""
    trace_id: str
    signal_id: str
    symbol: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[TraceStep] = field(default_factory=list)
    total_latency_ms: Optional[float] = None
    outcome: Optional[str] = None  # "executed", "rejected", "timeout"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceAnalysis:
    """Analysis of a decision trace"""
    trace_id: str
    total_latency_ms: float
    num_steps: int

    # Bottlenecks
    slowest_step: str
    slowest_step_latency_ms: float

    # Breakdown
    step_breakdown: Dict[str, float]  # step_name -> latency_ms
    step_percentages: Dict[str, float]  # step_name -> % of total

    # Timeline
    timeline: str  # Human-readable timeline

    # Recommendations
    bottlenecks: List[str]
    recommendations: List[str]


class DecisionTracer:
    """
    Track decision timelines with detailed timing.

    Useful for:
    - Identifying slow gates
    - Optimizing execution flow
    - Debugging latency issues
    """

    def __init__(self):
        """Initialize decision tracer"""
        self.traces: Dict[str, DecisionTrace] = {}
        self._trace_counter = 0

        logger.info("decision_tracer_initialized")

    def start_trace(
        self,
        signal_id: str,
        symbol: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new decision trace.

        Args:
            signal_id: Signal ID
            symbol: Trading symbol
            metadata: Optional metadata

        Returns:
            trace_id
        """
        self._trace_counter += 1
        trace_id = f"trace_{self._trace_counter:06d}"

        trace = DecisionTrace(
            trace_id=trace_id,
            signal_id=signal_id,
            symbol=symbol,
            start_time=time.time() * 1000,  # milliseconds
            metadata=metadata or {}
        )

        self.traces[trace_id] = trace

        logger.debug(
            "trace_started",
            trace_id=trace_id,
            signal_id=signal_id,
            symbol=symbol
        )

        return trace_id

    def record_step(
        self,
        trace_id: str,
        step_name: str,
        latency_ms: float,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a step in the trace.

        Args:
            trace_id: Trace ID
            step_name: Name of step
            latency_ms: Latency in milliseconds
            result: Result ("PASS", "FAIL", etc.)
            metadata: Optional metadata
        """
        if trace_id not in self.traces:
            logger.warning("trace_not_found", trace_id=trace_id)
            return

        trace = self.traces[trace_id]

        # Compute timestamps
        if trace.steps:
            start_time_ms = trace.steps[-1].end_time_ms
        else:
            start_time_ms = trace.start_time

        end_time_ms = start_time_ms + latency_ms

        step = TraceStep(
            step_name=step_name,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            latency_ms=latency_ms,
            result=result,
            metadata=metadata or {}
        )

        trace.steps.append(step)

        logger.debug(
            "trace_step_recorded",
            trace_id=trace_id,
            step_name=step_name,
            latency_ms=latency_ms,
            result=result
        )

    def finish_trace(
        self,
        trace_id: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Finish a trace.

        Args:
            trace_id: Trace ID
            outcome: "executed", "rejected", "timeout"
            metadata: Optional metadata
        """
        if trace_id not in self.traces:
            logger.warning("trace_not_found", trace_id=trace_id)
            return

        trace = self.traces[trace_id]
        trace.end_time = time.time() * 1000
        trace.outcome = outcome

        if metadata:
            trace.metadata.update(metadata)

        # Compute total latency
        if trace.steps:
            trace.total_latency_ms = trace.steps[-1].end_time_ms - trace.start_time
        else:
            trace.total_latency_ms = trace.end_time - trace.start_time

        logger.info(
            "trace_finished",
            trace_id=trace_id,
            outcome=outcome,
            total_latency_ms=trace.total_latency_ms,
            num_steps=len(trace.steps)
        )

    def get_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        """Get a trace by ID"""
        return self.traces.get(trace_id)

    def analyze_trace(self, trace_id: str) -> Optional[TraceAnalysis]:
        """
        Analyze a trace and identify bottlenecks.

        Args:
            trace_id: Trace ID

        Returns:
            TraceAnalysis with timing breakdown and recommendations
        """
        trace = self.get_trace(trace_id)
        if not trace:
            return None

        if not trace.steps:
            return None

        # Compute breakdown
        step_breakdown = {}
        for step in trace.steps:
            if step.step_name in step_breakdown:
                step_breakdown[step.step_name] += step.latency_ms
            else:
                step_breakdown[step.step_name] = step.latency_ms

        total_latency = trace.total_latency_ms or sum(step_breakdown.values())

        # Compute percentages
        step_percentages = {
            name: (latency / total_latency * 100) if total_latency > 0 else 0
            for name, latency in step_breakdown.items()
        }

        # Find slowest step
        slowest_step = max(step_breakdown.items(), key=lambda x: x[1])
        slowest_step_name = slowest_step[0]
        slowest_step_latency = slowest_step[1]

        # Generate timeline
        timeline = self._generate_timeline(trace)

        # Identify bottlenecks (>20% of total time)
        bottlenecks = [
            f"{name} ({latency:.1f}ms, {pct:.1f}%)"
            for name, latency in step_breakdown.items()
            if (pct := step_percentages[name]) > 20
        ]

        # Generate recommendations
        recommendations = []

        if slowest_step_latency > 5.0:
            recommendations.append(
                f"Optimize {slowest_step_name} - taking {slowest_step_latency:.1f}ms "
                f"({step_percentages[slowest_step_name]:.1f}% of total)"
            )

        # Check for gate bottlenecks
        gate_latency = sum(
            latency for name, latency in step_breakdown.items()
            if 'gate' in name.lower()
        )
        if gate_latency > total_latency * 0.5:
            recommendations.append(
                f"Gates taking {gate_latency:.1f}ms ({gate_latency/total_latency*100:.1f}% of total) - "
                "consider parallelization"
            )

        # Check for execution latency
        exec_latency = sum(
            latency for name, latency in step_breakdown.items()
            if 'exec' in name.lower() or 'trade' in name.lower()
        )
        if exec_latency > 50.0:
            recommendations.append(
                f"Execution taking {exec_latency:.1f}ms - check exchange API latency"
            )

        if not recommendations:
            recommendations.append("No major bottlenecks detected - system performing well")

        return TraceAnalysis(
            trace_id=trace_id,
            total_latency_ms=total_latency,
            num_steps=len(trace.steps),
            slowest_step=slowest_step_name,
            slowest_step_latency_ms=slowest_step_latency,
            step_breakdown=step_breakdown,
            step_percentages=step_percentages,
            timeline=timeline,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )

    def _generate_timeline(self, trace: DecisionTrace) -> str:
        """Generate human-readable timeline"""
        lines = [
            f"Decision Timeline for {trace.signal_id} ({trace.symbol})",
            "=" * 80
        ]

        cumulative_ms = 0.0
        for step in trace.steps:
            result_str = f" → {step.result}" if step.result else ""
            lines.append(
                f"  {cumulative_ms:6.1f}ms: {step.step_name} "
                f"({step.latency_ms:.1f}ms){result_str}"
            )
            cumulative_ms += step.latency_ms

        lines.append(f"  {'':->78}")
        lines.append(f"  TOTAL: {cumulative_ms:.1f}ms → {trace.outcome or 'UNKNOWN'}")

        return "\n".join(lines)

    def get_aggregate_stats(self, last_n: int = 100) -> Dict[str, Any]:
        """
        Get aggregate statistics across recent traces.

        Args:
            last_n: Number of recent traces to analyze

        Returns:
            Dict with average latencies, bottleneck frequency, etc.
        """
        recent_traces = list(self.traces.values())[-last_n:]

        if not recent_traces:
            return {}

        # Aggregate step latencies
        step_latencies: Dict[str, List[float]] = {}
        for trace in recent_traces:
            for step in trace.steps:
                if step.step_name not in step_latencies:
                    step_latencies[step.step_name] = []
                step_latencies[step.step_name].append(step.latency_ms)

        # Compute averages
        avg_latencies = {
            name: sum(latencies) / len(latencies)
            for name, latencies in step_latencies.items()
        }

        # Compute p95
        p95_latencies = {}
        for name, latencies in step_latencies.items():
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95_latencies[name] = sorted_latencies[p95_idx]

        # Total latencies
        total_latencies = [
            trace.total_latency_ms for trace in recent_traces
            if trace.total_latency_ms is not None
        ]

        avg_total = sum(total_latencies) / len(total_latencies) if total_latencies else 0

        return {
            "num_traces": len(recent_traces),
            "avg_total_latency_ms": avg_total,
            "avg_step_latencies": avg_latencies,
            "p95_step_latencies": p95_latencies,
            "slowest_step_avg": max(avg_latencies.items(), key=lambda x: x[1]) if avg_latencies else None
        }


if __name__ == '__main__':
    # Example usage
    print("Decision Trace Example")
    print("=" * 80)

    tracer = DecisionTracer()

    # Simulate decision flow
    print("\n📊 Simulating decision flow...")

    trace_id = tracer.start_trace(
        signal_id="sig_12345",
        symbol="ETH-USD",
        metadata={"price": 2045.50, "mode": "scalp"}
    )

    # Record steps
    tracer.record_step(trace_id, "signal_received", latency_ms=0.1)
    tracer.record_step(trace_id, "meta_label_gate", latency_ms=2.3, result="PASS")
    tracer.record_step(trace_id, "cost_gate", latency_ms=0.8, result="PASS")
    tracer.record_step(trace_id, "confidence_gate", latency_ms=0.5, result="PASS")
    tracer.record_step(trace_id, "regime_gate", latency_ms=0.3, result="PASS")
    tracer.record_step(trace_id, "spread_gate", latency_ms=0.4, result="PASS")
    tracer.record_step(trace_id, "volume_gate", latency_ms=0.3, result="PASS")
    tracer.record_step(trace_id, "trade_execution", latency_ms=45.0, result="SUCCESS")

    tracer.finish_trace(trace_id, outcome="executed")

    print(f"✓ Trace recorded: {trace_id}")

    # Analyze
    print("\n📈 Analyzing trace...")
    analysis = tracer.analyze_trace(trace_id)

    if analysis:
        print(f"\n{analysis.timeline}")

        print(f"\n📊 Breakdown:")
        for step, latency in sorted(
            analysis.step_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = analysis.step_percentages[step]
            print(f"  {step:30s} {latency:6.1f}ms ({pct:5.1f}%)")

        print(f"\n🔍 Analysis:")
        print(f"  Slowest step: {analysis.slowest_step} ({analysis.slowest_step_latency_ms:.1f}ms)")

        if analysis.bottlenecks:
            print(f"\n⚠️  Bottlenecks:")
            for bottleneck in analysis.bottlenecks:
                print(f"  • {bottleneck}")

        print(f"\n💡 Recommendations:")
        for rec in analysis.recommendations:
            print(f"  • {rec}")

    # Aggregate stats
    print("\n📊 Aggregate statistics (simulating 10 traces)...")

    # Simulate more traces
    for i in range(9):
        tid = tracer.start_trace(f"sig_{i}", "ETH-USD")
        tracer.record_step(tid, "meta_label_gate", latency_ms=2.0 + i * 0.1)
        tracer.record_step(tid, "cost_gate", latency_ms=0.8)
        tracer.record_step(tid, "trade_execution", latency_ms=45.0 + i * 2.0)
        tracer.finish_trace(tid, outcome="executed")

    stats = tracer.get_aggregate_stats(last_n=10)
    print(f"\n  Total traces: {stats['num_traces']}")
    print(f"  Average total latency: {stats['avg_total_latency_ms']:.1f}ms")
    print(f"\n  Average step latencies:")
    for step, latency in sorted(
        stats['avg_step_latencies'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"    {step:30s} {latency:6.1f}ms")

    if stats['slowest_step_avg']:
        slowest = stats['slowest_step_avg']
        print(f"\n  Slowest step (avg): {slowest[0]} ({slowest[1]:.1f}ms)")

    print("\n✓ Decision tracer ready!")
