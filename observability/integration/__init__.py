"""
Integration Module

Easy-to-use hooks for integrating observability into simulations.

Quick Start:
    from observability.integration import ObservabilityHooks

    hooks = ObservabilityHooks()
    hooks.on_signal(...)
    hooks.on_trade(...)
    summary = hooks.get_daily_summary()
"""

from observability.integration.hooks import ObservabilityHooks, get_hooks, reset_hooks

__all__ = ['ObservabilityHooks', 'get_hooks', 'reset_hooks']
