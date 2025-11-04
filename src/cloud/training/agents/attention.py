"""
Attention Mechanisms for Pattern Weighting

Implements attention-based pattern recognition and weighting for the RL agent.

Traditional approach: All features weighted equally
Attention approach: Learn which features/patterns matter most in each context

Key components:
1. Self-Attention: Which features correlate with each other?
2. Temporal Attention: Which time periods are most relevant?
3. Pattern Attention: Which historical patterns match current state?

Benefits:
- Dynamic feature importance (not fixed)
- Long-range dependencies (look far back when useful)
- Interpretability (see what agent focuses on)
- Performance (focus on signal, ignore noise)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AttentionConfig:
    """Attention mechanism configuration."""

    num_heads: int = 8  # Multi-head attention
    key_dim: int = 64  # Dimension of key/query/value
    dropout: float = 0.1  # Dropout rate
    max_sequence_length: int = 100  # Maximum lookback
    enable_self_attention: bool = True  # Self-attention on features
    enable_temporal_attention: bool = True  # Attention over time
    enable_pattern_attention: bool = True  # Attention over patterns


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Allows the model to attend to different aspects simultaneously.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Attention mask [batch, seq_len, seq_len]

        Returns:
            (output, attention_weights)
        """
        batch_size = query.size(0)

        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        Q = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back to [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output, attn_weights


class SelfAttentionFeatureEncoder(nn.Module):
    """
    Self-attention over features.

    Learns which features are relevant together (e.g., RSI + momentum + volume).
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention to features.

        Args:
            features: Feature tensor [batch, feature_dim]

        Returns:
            (attended_features, attention_weights)
        """
        # Reshape to [batch, 1, feature_dim] for self-attention
        x = features.unsqueeze(1)

        # Self-attention with residual
        attn_output, attn_weights = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        # Squeeze back to [batch, feature_dim]
        output = x.squeeze(1)

        return output, attn_weights


class TemporalAttentionEncoder(nn.Module):
    """
    Temporal attention over historical states.

    Learns which past timesteps are most relevant for current decision.
    Example: Recent spike in volume 5 candles ago is more important than
             noise 50 candles ago.
    """

    def __init__(
        self,
        state_dim: int,
        num_heads: int = 8,
        max_sequence_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.max_sequence_length = max_sequence_length

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_sequence_length, state_dim)
        )
        nn.init.normal_(self.positional_encoding, std=0.02)

        # Temporal attention
        self.temporal_attention = MultiHeadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 4, state_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state_sequence: torch.Tensor,
        current_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.

        Args:
            state_sequence: Historical states [batch, seq_len, state_dim]
            current_state: Current state [batch, state_dim]

        Returns:
            (context_vector, attention_weights)
        """
        batch_size, seq_len, _ = state_sequence.size()

        # Add positional encoding
        x = state_sequence + self.positional_encoding[:, :seq_len, :]

        # Current state as query, history as key/value
        query = current_state.unsqueeze(1)  # [batch, 1, state_dim]

        # Temporal attention
        attn_output, attn_weights = self.temporal_attention(query, x, x)
        attended = self.norm1(query + self.dropout(attn_output))

        # Feed-forward
        ffn_output = self.ffn(attended)
        output = self.norm2(attended + self.dropout(ffn_output))

        # Squeeze to [batch, state_dim]
        context = output.squeeze(1)

        return context, attn_weights


class PatternAttentionMatcher(nn.Module):
    """
    Pattern attention over memory bank.

    Retrieves and weights similar historical patterns.
    Example: Current state looks like previous breakout → recall breakout patterns.
    """

    def __init__(
        self,
        state_dim: int,
        pattern_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.pattern_dim = pattern_dim

        # Encode states to pattern space
        self.pattern_encoder = nn.Sequential(
            nn.Linear(state_dim, pattern_dim),
            nn.ReLU(),
            nn.Linear(pattern_dim, pattern_dim),
        )

        # Cross-attention (current state attends to memory)
        self.pattern_attention = MultiHeadAttention(
            embed_dim=pattern_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Combine current state with retrieved patterns
        self.fusion = nn.Sequential(
            nn.Linear(pattern_dim * 2, pattern_dim),
            nn.ReLU(),
            nn.Linear(pattern_dim, state_dim),
        )

    def forward(
        self,
        current_state: torch.Tensor,
        memory_patterns: torch.Tensor,
        memory_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve and weight similar patterns.

        Args:
            current_state: Current state [batch, state_dim]
            memory_patterns: Historical patterns [batch, num_patterns, state_dim]
            memory_rewards: Rewards from patterns [batch, num_patterns]

        Returns:
            (enriched_state, attention_weights, weighted_rewards)
        """
        batch_size = current_state.size(0)

        # Encode current state to pattern space
        query_pattern = self.pattern_encoder(current_state).unsqueeze(1)  # [batch, 1, pattern_dim]

        # Encode memory patterns
        num_patterns = memory_patterns.size(1)
        memory_encoded = self.pattern_encoder(
            memory_patterns.view(-1, self.state_dim)
        ).view(batch_size, num_patterns, self.pattern_dim)

        # Attention over memory
        attended_memory, attn_weights = self.pattern_attention(
            query_pattern,
            memory_encoded,
            memory_encoded,
        )

        # Compute weighted rewards
        # attn_weights: [batch, num_heads, 1, num_patterns]
        # Average over heads and squeeze
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)  # [batch, num_patterns]
        weighted_rewards = (avg_attn_weights * memory_rewards).sum(dim=-1)  # [batch]

        # Fuse current state with retrieved patterns
        combined = torch.cat([query_pattern.squeeze(1), attended_memory.squeeze(1)], dim=-1)
        enriched = self.fusion(combined)

        return enriched, avg_attn_weights, weighted_rewards


class AttentionAugmentedAgent(nn.Module):
    """
    Complete attention-augmented agent.

    Combines all attention mechanisms for powerful pattern recognition.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: AttentionConfig,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Self-attention over features
        if config.enable_self_attention:
            self.feature_attention = SelfAttentionFeatureEncoder(
                feature_dim=state_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )

        # Temporal attention over history
        if config.enable_temporal_attention:
            self.temporal_attention = TemporalAttentionEncoder(
                state_dim=state_dim,
                num_heads=config.num_heads,
                max_sequence_length=config.max_sequence_length,
                dropout=config.dropout,
            )

        # Pattern attention over memory
        if config.enable_pattern_attention:
            self.pattern_attention = PatternAttentionMatcher(
                state_dim=state_dim,
                pattern_dim=128,
                num_heads=config.num_heads // 2,
                dropout=config.dropout,
            )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        logger.info(
            "attention_augmented_agent_initialized",
            self_attention=config.enable_self_attention,
            temporal_attention=config.enable_temporal_attention,
            pattern_attention=config.enable_pattern_attention,
        )

    def forward(
        self,
        current_state: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        memory_patterns: Optional[torch.Tensor] = None,
        memory_rewards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with attention.

        Args:
            current_state: Current state [batch, state_dim]
            state_history: Historical states [batch, seq_len, state_dim]
            memory_patterns: Memory bank patterns [batch, num_patterns, state_dim]
            memory_rewards: Rewards from patterns [batch, num_patterns]

        Returns:
            (action_logits, value, attention_info)
        """
        attention_info = {}
        x = current_state

        # Self-attention over features
        if self.config.enable_self_attention:
            x, feature_attn = self.feature_attention(x)
            attention_info["feature_attention"] = feature_attn

        # Temporal attention over history
        if self.config.enable_temporal_attention and state_history is not None:
            temporal_context, temporal_attn = self.temporal_attention(state_history, x)
            x = x + temporal_context  # Residual connection
            attention_info["temporal_attention"] = temporal_attn

        # Pattern attention over memory
        if (
            self.config.enable_pattern_attention
            and memory_patterns is not None
            and memory_rewards is not None
        ):
            pattern_context, pattern_attn, weighted_reward = self.pattern_attention(
                x, memory_patterns, memory_rewards
            )
            x = x + pattern_context  # Residual connection
            attention_info["pattern_attention"] = pattern_attn
            attention_info["weighted_reward"] = weighted_reward

        # Policy and value
        action_logits = self.policy_head(x)
        value = self.value_head(x)

        return action_logits, value, attention_info

    def get_attention_weights(self, attention_info: Dict) -> Dict[str, torch.Tensor]:
        """Extract attention weights for visualization."""
        return {
            "feature": attention_info.get("feature_attention"),
            "temporal": attention_info.get("temporal_attention"),
            "pattern": attention_info.get("pattern_attention"),
        }
