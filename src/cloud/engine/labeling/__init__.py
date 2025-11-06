"""
Labeling System for Huracan V2

Proper labeling is CRITICAL for model quality. This system implements:

1. **Triple-Barrier Method**: For each potential entry, set TP/SL/timeout barriers
   and find which hits first. This prevents lookahead bias.

2. **Meta-Labeling**: After computing P&L, ask "was this profitable AFTER costs?"
   This trains the model to only take trades that beat transaction costs.

3. **Horizon-Specific Labels**: Different labels for scalp vs runner modes
   because they have different objectives and timeframes.

Without proper labeling, your model will learn from:
- Lookahead bias (knowing the future)
- Unrealistic P&L (ignoring costs)
- Mixed objectives (scalp + runner confused)
"""

from .label_schemas import (
    LabelConfig,
    LabeledTrade,
    ScalpLabelConfig,
    RunnerLabelConfig
)
from .meta_labeler import MetaLabeler
from .triple_barrier import TripleBarrierLabeler

__all__ = [
    'TripleBarrierLabeler',
    'MetaLabeler',
    'LabelConfig',
    'ScalpLabelConfig',
    'RunnerLabelConfig',
    'LabeledTrade',
]
