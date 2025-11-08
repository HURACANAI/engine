"""
Kelly Criterion

Calculate optimal position sizing using Kelly formula.
"""

import numpy as np


def calculate_kelly_fraction(
    win_probability: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly fraction for position sizing

    Kelly formula: f = (p * b - q) / b
    where:
      - p = win probability
      - q = loss probability (1 - p)
      - b = win/loss ratio (avg_win / avg_loss)

    Args:
        win_probability: Probability of winning trade [0-1]
        avg_win: Average win size (positive)
        avg_loss: Average loss size (positive)

    Returns:
        Kelly fraction [0-1]

    Example:
        kelly = calculate_kelly_fraction(
            win_probability=0.55,
            avg_win=100,
            avg_loss=80
        )
        # Position size = kelly * account_balance
    """
    if avg_loss == 0:
        return 0.0

    p = win_probability
    q = 1 - p
    b = avg_win / avg_loss

    kelly = (p * b - q) / b

    # Clip to [0, 1]
    kelly = max(0.0, min(1.0, kelly))

    return kelly


def calculate_half_kelly(
    win_probability: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate half-Kelly (more conservative)

    Half-Kelly reduces position size by 50% for safety.

    Args:
        win_probability: Win probability
        avg_win: Average win
        avg_loss: Average loss

    Returns:
        Half-Kelly fraction
    """
    full_kelly = calculate_kelly_fraction(win_probability, avg_win, avg_loss)
    return full_kelly * 0.5


def calculate_fractional_kelly(
    win_probability: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25
) -> float:
    """
    Calculate fractional Kelly

    Args:
        win_probability: Win probability
        avg_win: Average win
        avg_loss: Average loss
        fraction: Kelly fraction to use (0.25 = quarter Kelly)

    Returns:
        Fractional Kelly
    """
    full_kelly = calculate_kelly_fraction(win_probability, avg_win, avg_loss)
    return full_kelly * fraction


def calculate_kelly_with_uncertainty(
    win_probability: float,
    win_probability_std: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly with uncertainty in win probability

    Uses more conservative estimate when uncertainty is high.

    Args:
        win_probability: Mean win probability
        win_probability_std: Std of win probability
        avg_win: Average win
        avg_loss: Average loss

    Returns:
        Conservative Kelly fraction
    """
    # Use lower bound of confidence interval (mean - 2*std)
    conservative_p = max(0.0, win_probability - 2 * win_probability_std)

    kelly = calculate_kelly_fraction(conservative_p, avg_win, avg_loss)

    return kelly
