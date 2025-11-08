"""
Operators for scan conditions.

This module implements the operators used in scan conditions
(comparisons, crosses, patterns, etc.).
"""

import pandas as pd
import numpy as np
from typing import Union, Any


def compare(
    left: Union[pd.Series, float],
    operator: str,
    right: Union[pd.Series, float]
) -> pd.Series:
    """
    Compare two values or series.

    Args:
        left: Left operand
        operator: Comparison operator (>, <, >=, <=, ==, !=)
        right: Right operand

    Returns:
        Boolean Series

    Examples:
        >>> result = compare(df['close'], '>', 100)
        >>> result = compare(df['rsi'], '<', 30)
    """
    # Ensure we have Series
    if not isinstance(left, pd.Series):
        if isinstance(right, pd.Series):
            left = pd.Series([left] * len(right), index=right.index)
        else:
            raise ValueError("At least one operand must be a Series")

    if not isinstance(right, pd.Series):
        right = pd.Series([right] * len(left), index=left.index)

    # Perform comparison
    if operator == '>':
        return left > right
    elif operator == '<':
        return left < right
    elif operator == '>=':
        return left >= right
    elif operator == '<=':
        return left <= right
    elif operator == '==':
        return left == right
    elif operator == '!=':
        return left != right
    else:
        raise ValueError(f"Unknown operator: {operator}")


def crosses_above(
    series1: pd.Series,
    series2: pd.Series,
    lookback: int = 5
) -> pd.Series:
    """
    Detect where series1 crosses above series2.

    Args:
        series1: First series
        series2: Second series
        lookback: Number of bars to check for cross

    Returns:
        Boolean Series indicating crossover points

    Examples:
        >>> crossed = crosses_above(df['ema_fast'], df['ema_slow'])
    """
    result = pd.Series(False, index=series1.index)

    for i in range(lookback, len(series1)):
        # Check if currently above
        if series1.iloc[i] > series2.iloc[i]:
            # Check if was below in recent past
            was_below = False
            for j in range(1, lookback + 1):
                if i - j >= 0 and series1.iloc[i - j] < series2.iloc[i - j]:
                    was_below = True
                    break

            if was_below:
                result.iloc[i] = True

    return result


def crosses_below(
    series1: pd.Series,
    series2: pd.Series,
    lookback: int = 5
) -> pd.Series:
    """
    Detect where series1 crosses below series2.

    Args:
        series1: First series
        series2: Second series
        lookback: Number of bars to check for cross

    Returns:
        Boolean Series indicating crossunder points

    Examples:
        >>> crossed = crosses_below(df['ema_fast'], df['ema_slow'])
    """
    result = pd.Series(False, index=series1.index)

    for i in range(lookback, len(series1)):
        # Check if currently below
        if series1.iloc[i] < series2.iloc[i]:
            # Check if was above in recent past
            was_above = False
            for j in range(1, lookback + 1):
                if i - j >= 0 and series1.iloc[i - j] > series2.iloc[i - j]:
                    was_above = True
                    break

            if was_above:
                result.iloc[i] = True

    return result


def logical_and(conditions: list) -> pd.Series:
    """
    Combine conditions with AND logic.

    Args:
        conditions: List of boolean Series

    Returns:
        Boolean Series (True where all conditions are True)

    Examples:
        >>> result = logical_and([cond1, cond2, cond3])
    """
    if not conditions:
        return pd.Series(True)

    result = conditions[0]
    for cond in conditions[1:]:
        result = result & cond

    return result


def logical_or(conditions: list) -> pd.Series:
    """
    Combine conditions with OR logic.

    Args:
        conditions: List of boolean Series

    Returns:
        Boolean Series (True where any condition is True)

    Examples:
        >>> result = logical_or([cond1, cond2, cond3])
    """
    if not conditions:
        return pd.Series(False)

    result = conditions[0]
    for cond in conditions[1:]:
        result = result | cond

    return result


def check_pattern(
    df: pd.DataFrame,
    pattern_type: str
) -> pd.Series:
    """
    Check for candlestick pattern.

    Args:
        df: DataFrame with OHLCV data
        pattern_type: Pattern name

    Returns:
        Boolean Series

    Examples:
        >>> has_doji = check_pattern(df, 'doji')
    """
    # This would integrate with the candlesticks module
    from src.candlesticks import CandlestickDetector

    detector = CandlestickDetector()

    try:
        pattern_series = detector.detect(df, pattern_type)
        # Convert to boolean (non-zero = pattern detected)
        return pattern_series != 0
    except Exception:
        # Pattern not found or error
        return pd.Series(False, index=df.index)


def get_value(
    df: pd.DataFrame,
    column: str,
    default: Any = None
) -> Union[pd.Series, Any]:
    """
    Safely get value from DataFrame.

    Args:
        df: DataFrame
        column: Column name
        default: Default value if column doesn't exist

    Returns:
        Series or default value

    Examples:
        >>> close = get_value(df, 'close')
        >>> rsi = get_value(df, 'rsi', default=50)
    """
    if column in df.columns:
        return df[column]
    elif default is not None:
        return default
    else:
        raise KeyError(f"Column '{column}' not found in DataFrame")


def recent_high(series: pd.Series, period: int = 20) -> pd.Series:
    """Get rolling maximum."""
    return series.rolling(window=period).max()


def recent_low(series: pd.Series, period: int = 20) -> pd.Series:
    """Get rolling minimum."""
    return series.rolling(window=period).min()


def percent_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percent change."""
    return series.pct_change(periods=periods) * 100


def is_increasing(series: pd.Series, periods: int = 5) -> pd.Series:
    """Check if series is increasing over periods."""
    return series > series.shift(periods)


def is_decreasing(series: pd.Series, periods: int = 5) -> pd.Series:
    """Check if series is decreasing over periods."""
    return series < series.shift(periods)
