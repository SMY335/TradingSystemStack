"""
Trendline Detection Module.

This module provides automatic detection of support and resistance trendlines,
as well as breakout detection capabilities.

Examples:
    >>> from src.trendlines import detect_support_resistance, detect_breakouts
    >>> import pandas as pd
    >>>
    >>> # Detect trendlines
    >>> trendlines = detect_support_resistance(df, min_touches=3)
    >>> supports = [t for t in trendlines if t.type == 'support']
    >>> resistances = [t for t in trendlines if t.type == 'resistance']
    >>>
    >>> # Detect breakouts
    >>> breakouts = detect_breakouts(df, trendlines)
    >>> bullish = [b for b in breakouts if b.is_bullish()]
"""

# Main detector classes
from .detector import (
    TrendlineDetector,
    Trendline,
    detect_support_resistance
)

# Breakout detection
from .breakout import (
    BreakoutDetector,
    Breakout,
    detect_breakouts
)

# Algorithms
from .algorithms import (
    TrendlineCandidate,
    detect_swing_points,
    fit_regression_line,
    find_trendline_candidates,
    merge_similar_trendlines,
    calculate_trendline_strength,
    filter_trendlines_by_recency,
    extend_trendline
)

__all__ = [
    # Main API
    'TrendlineDetector',
    'Trendline',
    'detect_support_resistance',

    # Breakout detection
    'BreakoutDetector',
    'Breakout',
    'detect_breakouts',

    # Algorithms
    'TrendlineCandidate',
    'detect_swing_points',
    'fit_regression_line',
    'find_trendline_candidates',
    'merge_similar_trendlines',
    'calculate_trendline_strength',
    'filter_trendlines_by_recency',
    'extend_trendline',
]

__version__ = '2.0.0'
