"""
Sample Weighting for Huracan V2

Recency weighting gives higher importance to recent samples because:
1. Market microstructure changes (new algos, fee updates, venue changes)
2. Recent patterns more predictive of tomorrow
3. But don't discard old data entirely (regime memory)

Exponential decay formula: weight = exp(-λ × days_ago)
where λ = ln(2) / halflife

Example (halflife=10 days):
- Today: weight = 1.0
- 10 days ago: weight = 0.5
- 20 days ago: weight = 0.25
- 60 days ago: weight = 0.016
"""

from .recency_weighter import RecencyWeighter, create_mode_specific_weighter

__all__ = ['RecencyWeighter', 'create_mode_specific_weighter']
