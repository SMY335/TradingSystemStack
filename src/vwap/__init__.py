"""
Anchored VWAP (Volume Weighted Average Price) for TradingSystemStack.

Provides VWAP calculations anchored to specific points: session start,
week/month start, swing highs/lows, custom timestamps, or custom functions.

Examples:
    >>> from src.vwap import AnchoredVWAP, calculate_vwap
    >>>
    >>> # Using calculator class
    >>> vwap_calc = AnchoredVWAP(include_bands=True)
    >>> result = vwap_calc.calculate(df, anchor_type='session')
    >>>
    >>> # Using convenience function
    >>> vwap_df = calculate_vwap(df, anchor_type='week')
    >>> vwap_df = calculate_vwap(df, anchor_type='swing_high', lookback=20)
    >>>
    >>> # Anchor to specific timestamp
    >>> vwap_df = calculate_vwap_from_timestamp(df, '2024-01-01')
    >>>
    >>> # Multiple VWAPs at once
    >>> vwaps = calculate_multiple_vwaps(df, ['session', 'week', 'month'])
"""

from .anchored_vwap import (
    AnchoredVWAP,
    calculate_vwap,
    calculate_vwap_from_timestamp,
    calculate_multiple_vwaps,
)

from .anchors import (
    session_anchor,
    week_anchor,
    month_anchor,
    year_anchor,
    swing_high_anchor,
    swing_low_anchor,
    timestamp_anchor,
    custom_anchor,
    get_anchor,
    AnchorError,
    ANCHOR_TYPES,
)


__all__ = [
    # Main calculator
    'AnchoredVWAP',

    # Convenience functions
    'calculate_vwap',
    'calculate_vwap_from_timestamp',
    'calculate_multiple_vwaps',

    # Anchor functions
    'session_anchor',
    'week_anchor',
    'month_anchor',
    'year_anchor',
    'swing_high_anchor',
    'swing_low_anchor',
    'timestamp_anchor',
    'custom_anchor',
    'get_anchor',

    # Constants
    'ANCHOR_TYPES',

    # Exceptions
    'AnchorError',
]
