"""
Anchor point definitions for Anchored VWAP.

Provides predefined anchors and utilities for custom anchors.
"""
from typing import Union, Callable, Optional
from datetime import datetime, time
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AnchorError(Exception):
    """Custom exception for anchor operations."""
    pass


def session_anchor(df: pd.DataFrame, session_start: time = time(9, 30)) -> pd.Series:
    """Anchor at start of each trading session.

    Args:
        df: DataFrame with DatetimeIndex
        session_start: Time when session starts (default: 9:30 AM)

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = session_anchor(df)
        >>> anchors = session_anchor(df, session_start=time(8, 0))
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnchorError("DataFrame must have DatetimeIndex")

    # Mark first bar of each day
    dates = df.index.date
    is_first = pd.Series(False, index=df.index)
    is_first.iloc[0] = True  # First bar is always an anchor

    # Mark when date changes
    for i in range(1, len(dates)):
        if dates[i] != dates[i-1]:
            is_first.iloc[i] = True

    return is_first


def week_anchor(df: pd.DataFrame, week_start: int = 0) -> pd.Series:
    """Anchor at start of each week.

    Args:
        df: DataFrame with DatetimeIndex
        week_start: Day of week to start (0=Monday, 6=Sunday)

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = week_anchor(df)  # Monday
        >>> anchors = week_anchor(df, week_start=6)  # Sunday
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnchorError("DataFrame must have DatetimeIndex")

    # Mark first bar of each week
    weeks = df.index.isocalendar().week
    is_first = pd.Series(False, index=df.index)
    is_first.iloc[0] = True  # First bar is always an anchor

    # Mark when week changes
    for i in range(1, len(weeks)):
        if weeks[i] != weeks[i-1]:
            is_first.iloc[i] = True

    return is_first


def month_anchor(df: pd.DataFrame) -> pd.Series:
    """Anchor at start of each month.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = month_anchor(df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnchorError("DataFrame must have DatetimeIndex")

    # Mark first bar of each month
    months = df.index.month
    is_first = pd.Series(False, index=df.index)
    is_first.iloc[0] = True  # First bar is always an anchor

    # Mark when month changes
    for i in range(1, len(months)):
        if months[i] != months[i-1]:
            is_first.iloc[i] = True

    return is_first


def year_anchor(df: pd.DataFrame) -> pd.Series:
    """Anchor at start of each year.

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = year_anchor(df)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnchorError("DataFrame must have DatetimeIndex")

    # Mark first bar of each year
    years = df.index.year
    is_first = pd.Series(False, index=df.index)
    is_first.iloc[0] = True  # First bar is always an anchor

    # Mark when year changes
    for i in range(1, len(years)):
        if years[i] != years[i-1]:
            is_first.iloc[i] = True

    return is_first


def swing_high_anchor(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.02
) -> pd.Series:
    """Anchor at swing highs.

    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for swing detection
        threshold: Minimum % move to qualify as swing (0.02 = 2%)

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = swing_high_anchor(df)
        >>> anchors = swing_high_anchor(df, lookback=10, threshold=0.01)
    """
    if 'high' not in df.columns:
        raise AnchorError("DataFrame must have 'high' column")

    highs = df['high']

    # Find local maxima
    is_swing = pd.Series(False, index=df.index)

    for i in range(lookback, len(df) - lookback):
        current_high = highs.iloc[i]
        lookback_max = highs.iloc[i-lookback:i].max()
        lookforward_max = highs.iloc[i+1:i+lookback+1].max()

        # Current is higher than lookback/lookforward
        if current_high > lookback_max and current_high > lookforward_max:
            # Check minimum threshold
            pct_move = (current_high - lookback_max) / lookback_max
            if pct_move >= threshold:
                is_swing.iloc[i] = True

    return is_swing


def swing_low_anchor(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.02
) -> pd.Series:
    """Anchor at swing lows.

    Args:
        df: DataFrame with OHLC data
        lookback: Lookback period for swing detection
        threshold: Minimum % move to qualify as swing (0.02 = 2%)

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = swing_low_anchor(df)
        >>> anchors = swing_low_anchor(df, lookback=10, threshold=0.01)
    """
    if 'low' not in df.columns:
        raise AnchorError("DataFrame must have 'low' column")

    lows = df['low']

    # Find local minima
    is_swing = pd.Series(False, index=df.index)

    for i in range(lookback, len(df) - lookback):
        current_low = lows.iloc[i]
        lookback_min = lows.iloc[i-lookback:i].min()
        lookforward_min = lows.iloc[i+1:i+lookback+1].min()

        # Current is lower than lookback/lookforward
        if current_low < lookback_min and current_low < lookforward_min:
            # Check minimum threshold
            pct_move = (lookback_min - current_low) / lookback_min
            if pct_move >= threshold:
                is_swing.iloc[i] = True

    return is_swing


def timestamp_anchor(
    df: pd.DataFrame,
    timestamp: Union[str, datetime]
) -> pd.Series:
    """Anchor at specific timestamp.

    Args:
        df: DataFrame with DatetimeIndex
        timestamp: Timestamp to anchor (or closest bar after)

    Returns:
        Boolean series marking anchor point

    Examples:
        >>> anchors = timestamp_anchor(df, '2024-01-01')
        >>> anchors = timestamp_anchor(df, datetime(2024, 1, 1, 9, 30))
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnchorError("DataFrame must have DatetimeIndex")

    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    # Find closest bar at or after timestamp
    anchor_series = pd.Series(False, index=df.index)

    if timestamp in df.index:
        anchor_series[timestamp] = True
    else:
        # Find first bar after timestamp
        future_bars = df.index[df.index >= timestamp]
        if len(future_bars) > 0:
            anchor_series[future_bars[0]] = True

    return anchor_series


def custom_anchor(
    df: pd.DataFrame,
    anchor_func: Callable[[pd.DataFrame], pd.Series]
) -> pd.Series:
    """Apply custom anchor function.

    Args:
        df: DataFrame with OHLC data
        anchor_func: Function that returns boolean series of anchors

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> # Anchor at gap ups > 2%
        >>> def gap_up_anchor(df):
        ...     gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        ...     return gaps > 0.02
        >>> anchors = custom_anchor(df, gap_up_anchor)
    """
    try:
        result = anchor_func(df)

        if not isinstance(result, pd.Series):
            raise AnchorError("Anchor function must return pd.Series")

        if len(result) != len(df):
            raise AnchorError("Anchor series must match DataFrame length")

        return result.astype(bool)

    except Exception as e:
        logger.error(f"Custom anchor function failed: {e}")
        raise AnchorError(f"Custom anchor failed: {e}") from e


# Predefined anchor types
ANCHOR_TYPES = {
    'session': session_anchor,
    'week': week_anchor,
    'month': month_anchor,
    'year': year_anchor,
    'swing_high': swing_high_anchor,
    'swing_low': swing_low_anchor,
}


def get_anchor(
    df: pd.DataFrame,
    anchor_type: str,
    **kwargs
) -> pd.Series:
    """Get anchor points by type.

    Args:
        df: DataFrame with OHLC data
        anchor_type: Type of anchor ('session', 'week', 'month', etc.)
        **kwargs: Additional arguments for anchor function

    Returns:
        Boolean series marking anchor points

    Examples:
        >>> anchors = get_anchor(df, 'session')
        >>> anchors = get_anchor(df, 'swing_high', lookback=15)
    """
    if anchor_type not in ANCHOR_TYPES:
        raise AnchorError(
            f"Unknown anchor type: {anchor_type}. "
            f"Available: {', '.join(ANCHOR_TYPES.keys())}"
        )

    anchor_func = ANCHOR_TYPES[anchor_type]
    return anchor_func(df, **kwargs)
