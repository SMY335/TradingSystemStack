"""
Pandas-based candlestick pattern detection (fallback without TA-Lib).

Implements common candlestick patterns using pandas operations.
"""
from typing import List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def detect_pattern_pandas_ta(
    df: pd.DataFrame,
    pattern: str,
    return_strength: bool = False
) -> pd.Series:
    """Detect pattern using pandas operations.

    Args:
        df: DataFrame with OHLC data
        pattern: Pattern name
        return_strength: Return pattern strength (default: False)

    Returns:
        Series with pattern signals (1=bullish, -1=bearish, 0=none)
    """
    pattern_lower = pattern.lower().replace(' ', '_')

    # Map pattern to detection function
    pattern_functions = {
        'doji': _detect_doji,
        'hammer': _detect_hammer,
        'inverted_hammer': _detect_inverted_hammer,
        'hanging_man': _detect_hanging_man,
        'shooting_star': _detect_shooting_star,
        'engulfing': _detect_engulfing,
        'harami': _detect_harami,
        'piercing': _detect_piercing,
        'dark_cloud_cover': _detect_dark_cloud_cover,
        'morning_star': _detect_morning_star,
        'evening_star': _detect_evening_star,
        'three_white_soldiers': _detect_three_white_soldiers,
        'three_black_crows': _detect_three_black_crows,
        'spinning_top': _detect_spinning_top,
        'marubozu': _detect_marubozu,
    }

    if pattern_lower not in pattern_functions:
        logger.warning(f"Pattern {pattern} not implemented in pandas fallback")
        return pd.Series(0, index=df.index)

    return pattern_functions[pattern_lower](df, return_strength)


def detect_all_pandas_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all implemented patterns using pandas.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with columns for each pattern
    """
    patterns = [
        'doji', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
        'engulfing', 'harami', 'piercing', 'dark_cloud_cover',
        'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
        'spinning_top', 'marubozu'
    ]

    results = {}
    for pattern in patterns:
        try:
            results[pattern] = detect_pattern_pandas_ta(df, pattern)
        except Exception as e:
            logger.warning(f"Failed to detect {pattern}: {e}")
            results[pattern] = pd.Series(0, index=df.index)

    return pd.DataFrame(results)


def get_available_patterns_pandas_ta() -> List[str]:
    """Get list of implemented patterns.

    Returns:
        List of pattern names
    """
    return [
        'doji', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
        'engulfing', 'harami', 'piercing', 'dark_cloud_cover',
        'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
        'spinning_top', 'marubozu'
    ]


# Helper functions
def _body_size(df: pd.DataFrame) -> pd.Series:
    """Calculate candle body size."""
    return (df['close'] - df['open']).abs()


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    """Calculate upper shadow size."""
    return df['high'] - df[['open', 'close']].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    """Calculate lower shadow size."""
    return df[['open', 'close']].min(axis=1) - df['low']


def _range(df: pd.DataFrame) -> pd.Series:
    """Calculate candle range (high - low)."""
    return df['high'] - df['low']


def _is_bullish(df: pd.DataFrame) -> pd.Series:
    """Check if candle is bullish (close > open)."""
    return df['close'] > df['open']


def _is_bearish(df: pd.DataFrame) -> pd.Series:
    """Check if candle is bearish (close < open)."""
    return df['close'] < df['open']


# Pattern detection functions
def _detect_doji(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Doji pattern.

    Criteria:
    - Body is very small compared to range
    - Upper and lower shadows are roughly equal
    """
    body = _body_size(df)
    candle_range = _range(df)

    # Body should be < 10% of range
    is_doji = body < (candle_range * 0.1)

    result = pd.Series(0, index=df.index)
    result[is_doji] = 1

    if return_strength:
        # Strength based on how small the body is
        strength = pd.Series(0, index=df.index)
        strength[is_doji] = ((candle_range - body) / candle_range * 100)[is_doji]
        return strength

    return result


def _detect_hammer(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Hammer pattern (bullish reversal).

    Criteria:
    - Small body at top of range
    - Long lower shadow (2x body)
    - Little to no upper shadow
    """
    body = _body_size(df)
    lower_shadow = _lower_shadow(df)
    upper_shadow = _upper_shadow(df)
    candle_range = _range(df)

    is_hammer = (
        (body < candle_range * 0.3) &  # Small body
        (lower_shadow > body * 2) &     # Long lower shadow
        (upper_shadow < body * 0.5)     # Small upper shadow
    )

    result = pd.Series(0, index=df.index)
    result[is_hammer] = 1

    if return_strength:
        strength = pd.Series(0, index=df.index)
        strength[is_hammer] = ((lower_shadow / body)[is_hammer]).clip(0, 100)
        return strength

    return result


def _detect_inverted_hammer(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Inverted Hammer pattern (bullish reversal).

    Criteria:
    - Small body at bottom of range
    - Long upper shadow (2x body)
    - Little to no lower shadow
    """
    body = _body_size(df)
    lower_shadow = _lower_shadow(df)
    upper_shadow = _upper_shadow(df)
    candle_range = _range(df)

    is_inv_hammer = (
        (body < candle_range * 0.3) &   # Small body
        (upper_shadow > body * 2) &     # Long upper shadow
        (lower_shadow < body * 0.5)     # Small lower shadow
    )

    result = pd.Series(0, index=df.index)
    result[is_inv_hammer] = 1

    return result


def _detect_hanging_man(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Hanging Man pattern (bearish reversal).

    Same shape as hammer but appears in uptrend.
    """
    # Same criteria as hammer but bearish
    hammer_result = _detect_hammer(df, return_strength)
    return -hammer_result if not return_strength else hammer_result


def _detect_shooting_star(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Shooting Star pattern (bearish reversal).

    Same shape as inverted hammer but appears in uptrend.
    """
    # Same criteria as inverted hammer but bearish
    inv_hammer_result = _detect_inverted_hammer(df, return_strength)
    return -inv_hammer_result if not return_strength else inv_hammer_result


def _detect_engulfing(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Engulfing pattern (bullish or bearish reversal).

    Criteria:
    - Second candle body completely engulfs first candle body
    - Opposite colors
    """
    result = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]

        # Bullish engulfing: prev bearish, curr bullish and engulfs
        if (prev_close < prev_open and  # Prev bearish
            curr_close > curr_open and  # Curr bullish
            curr_open < prev_close and  # Engulfs
            curr_close > prev_open):
            result.iloc[i] = 1

        # Bearish engulfing: prev bullish, curr bearish and engulfs
        elif (prev_close > prev_open and  # Prev bullish
              curr_close < curr_open and  # Curr bearish
              curr_open > prev_close and  # Engulfs
              curr_close < prev_open):
            result.iloc[i] = -1

    return result


def _detect_harami(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Harami pattern (reversal).

    Criteria:
    - Second candle body contained within first candle body
    - Opposite colors
    """
    result = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]

        prev_body_top = max(prev_open, prev_close)
        prev_body_bottom = min(prev_open, prev_close)
        curr_body_top = max(curr_open, curr_close)
        curr_body_bottom = min(curr_open, curr_close)

        # Current body contained within previous body
        if (curr_body_top < prev_body_top and
            curr_body_bottom > prev_body_bottom):

            # Bullish harami: prev bearish, curr bullish
            if prev_close < prev_open and curr_close > curr_open:
                result.iloc[i] = 1
            # Bearish harami: prev bullish, curr bearish
            elif prev_close > prev_open and curr_close < curr_open:
                result.iloc[i] = -1

    return result


def _detect_piercing(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Piercing pattern (bullish reversal).

    Criteria:
    - First candle bearish
    - Second candle opens below first close, closes above first midpoint
    """
    result = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]

        if (prev_close < prev_open and  # Prev bearish
            curr_close > curr_open and  # Curr bullish
            curr_open < prev_close and  # Opens below prev close
            curr_close > (prev_open + prev_close) / 2):  # Closes above midpoint
            result.iloc[i] = 1

    return result


def _detect_dark_cloud_cover(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Dark Cloud Cover pattern (bearish reversal).

    Criteria:
    - First candle bullish
    - Second candle opens above first close, closes below first midpoint
    """
    result = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        prev_open = df['open'].iloc[i-1]
        prev_close = df['close'].iloc[i-1]
        curr_open = df['open'].iloc[i]
        curr_close = df['close'].iloc[i]

        if (prev_close > prev_open and  # Prev bullish
            curr_close < curr_open and  # Curr bearish
            curr_open > prev_close and  # Opens above prev close
            curr_close < (prev_open + prev_close) / 2):  # Closes below midpoint
            result.iloc[i] = -1

    return result


def _detect_morning_star(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Morning Star pattern (bullish reversal).

    Criteria:
    - First candle bearish
    - Second candle small (doji or spinning top)
    - Third candle bullish, closes above first midpoint
    """
    result = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        first_open = df['open'].iloc[i-2]
        first_close = df['close'].iloc[i-2]
        second_range = _range(df).iloc[i-1]
        second_body = _body_size(df).iloc[i-1]
        third_open = df['open'].iloc[i]
        third_close = df['close'].iloc[i]

        if (first_close < first_open and  # First bearish
            second_body < second_range * 0.3 and  # Second small
            third_close > third_open and  # Third bullish
            third_close > (first_open + first_close) / 2):  # Closes above first midpoint
            result.iloc[i] = 1

    return result


def _detect_evening_star(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Evening Star pattern (bearish reversal).

    Criteria:
    - First candle bullish
    - Second candle small (doji or spinning top)
    - Third candle bearish, closes below first midpoint
    """
    result = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        first_open = df['open'].iloc[i-2]
        first_close = df['close'].iloc[i-2]
        second_range = _range(df).iloc[i-1]
        second_body = _body_size(df).iloc[i-1]
        third_open = df['open'].iloc[i]
        third_close = df['close'].iloc[i]

        if (first_close > first_open and  # First bullish
            second_body < second_range * 0.3 and  # Second small
            third_close < third_open and  # Third bearish
            third_close < (first_open + first_close) / 2):  # Closes below first midpoint
            result.iloc[i] = -1

    return result


def _detect_three_white_soldiers(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Three White Soldiers pattern (bullish continuation).

    Criteria:
    - Three consecutive bullish candles
    - Each opens within previous body
    - Each closes higher than previous
    """
    result = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        # All three bullish
        if all(df['close'].iloc[i-j] > df['open'].iloc[i-j] for j in range(3)):
            # Each opens within previous body
            if (df['open'].iloc[i-1] > df['open'].iloc[i-2] and
                df['open'].iloc[i-1] < df['close'].iloc[i-2] and
                df['open'].iloc[i] > df['open'].iloc[i-1] and
                df['open'].iloc[i] < df['close'].iloc[i-1]):
                # Each closes higher
                if (df['close'].iloc[i-1] > df['close'].iloc[i-2] and
                    df['close'].iloc[i] > df['close'].iloc[i-1]):
                    result.iloc[i] = 1

    return result


def _detect_three_black_crows(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Three Black Crows pattern (bearish continuation).

    Criteria:
    - Three consecutive bearish candles
    - Each opens within previous body
    - Each closes lower than previous
    """
    result = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        # All three bearish
        if all(df['close'].iloc[i-j] < df['open'].iloc[i-j] for j in range(3)):
            # Each opens within previous body
            if (df['open'].iloc[i-1] < df['open'].iloc[i-2] and
                df['open'].iloc[i-1] > df['close'].iloc[i-2] and
                df['open'].iloc[i] < df['open'].iloc[i-1] and
                df['open'].iloc[i] > df['close'].iloc[i-1]):
                # Each closes lower
                if (df['close'].iloc[i-1] < df['close'].iloc[i-2] and
                    df['close'].iloc[i] < df['close'].iloc[i-1]):
                    result.iloc[i] = -1

    return result


def _detect_spinning_top(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Spinning Top pattern (indecision).

    Criteria:
    - Small body (< 1/3 of range)
    - Upper and lower shadows approximately equal
    """
    body = _body_size(df)
    candle_range = _range(df)
    upper_shadow = _upper_shadow(df)
    lower_shadow = _lower_shadow(df)

    is_spinning = (
        (body < candle_range / 3) &
        (upper_shadow > body * 0.5) &
        (lower_shadow > body * 0.5)
    )

    result = pd.Series(0, index=df.index)
    result[is_spinning] = 1

    return result


def _detect_marubozu(df: pd.DataFrame, return_strength: bool = False) -> pd.Series:
    """Detect Marubozu pattern (strong trend).

    Criteria:
    - Body is > 90% of range (little to no shadows)
    """
    body = _body_size(df)
    candle_range = _range(df)

    is_marubozu = body > (candle_range * 0.9)

    result = pd.Series(0, index=df.index)
    # Bullish marubozu
    result[is_marubozu & _is_bullish(df)] = 1
    # Bearish marubozu
    result[is_marubozu & _is_bearish(df)] = -1

    return result
