"""
TA-Lib candlestick pattern detection.

Wraps TA-Lib's 60+ candlestick pattern recognition functions.
"""
from typing import List
import pandas as pd
import numpy as np
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)


# TA-Lib pattern function mappings
TALIB_PATTERNS = {
    # Single candlestick patterns
    'doji': 'CDLDOJI',
    'doji_star': 'CDLDOJISTAR',
    'dragonfly_doji': 'CDLDRAGONFLYDOJI',
    'gravestone_doji': 'CDLGRAVESTONEDOJI',
    'long_legged_doji': 'CDLLONGLEGGEDDOJI',
    'hammer': 'CDLHAMMER',
    'inverted_hammer': 'CDLINVERTEDHAMMER',
    'hanging_man': 'CDLHANGINGMAN',
    'shooting_star': 'CDLSHOOTINGSTAR',
    'marubozu': 'CDLMARUBOZU',
    'spinning_top': 'CDLSPINNINGTOP',
    'high_wave': 'CDLHIGHWAVE',

    # Two candlestick patterns
    'engulfing': 'CDLENGULFING',
    'harami': 'CDLHARAMI',
    'harami_cross': 'CDLHARAMICROSS',
    'piercing': 'CDLPIERCING',
    'dark_cloud_cover': 'CDLDARKCLOUDCOVER',
    'tweezer': 'CDL2CROWS',  # Approximation
    'kicking': 'CDLKICKING',
    'matching_low': 'CDLMATCHINGLOW',

    # Three candlestick patterns
    'morning_star': 'CDLMORNINGSTAR',
    'evening_star': 'CDLEVENINGSTAR',
    'morning_doji_star': 'CDLMORNINGDOJISTAR',
    'evening_doji_star': 'CDLEVENINGDOJISTAR',
    'three_white_soldiers': 'CDL3WHITESOLDIERS',
    'three_black_crows': 'CDL3BLACKCROWS',
    'three_inside_up': 'CDL3INSIDE',
    'three_inside_down': 'CDL3INSIDE',
    'three_outside_up': 'CDL3OUTSIDE',
    'three_outside_down': 'CDL3OUTSIDE',
    'three_line_strike': 'CDL3LINESTRIKE',
    'three_stars_in_south': 'CDL3STARSINSOUTH',
    'abandoned_baby': 'CDLABANDONEDBABY',
    'advance_block': 'CDLADVANCEBLOCK',
    'belt_hold': 'CDLBELTHOLD',
    'breakaway': 'CDLBREAKAWAY',
    'closing_marubozu': 'CDLCLOSINGMARUBOZU',
    'concealing_baby_swallow': 'CDLCONCEALBABYSWALL',
    'counterattack': 'CDLCOUNTERATTACK',
    'gap_sidesidewhite': 'CDLGAPSIDESIDEWHITE',
    'homing_pigeon': 'CDLHOMINGPIGEON',
    'identical_three_crows': 'CDLIDENTICAL3CROWS',
    'inneck': 'CDLINNECK',
    'kicking_by_length': 'CDLKICKINGBYLENGTH',
    'ladder_bottom': 'CDLLADDERBOTTOM',
    'mat_hold': 'CDLMATHOLD',
    'on_neck': 'CDLONNECK',
    'rickshaw_man': 'CDLRICKSHAWMAN',
    'rising_three_methods': 'CDLRISEFALL3METHODS',
    'falling_three_methods': 'CDLRISEFALL3METHODS',
    'separating_lines': 'CDLSEPARATINGLINES',
    'side_by_side_white': 'CDLSIDEBYSIDEWHITE',
    'stick_sandwich': 'CDLSTICKSANDWICH',
    'takuri': 'CDLTAKURI',
    'tasuki_gap': 'CDLTASUKIGAP',
    'thrusting': 'CDLTHRUSTING',
    'tristar': 'CDLTRISTAR',
    'two_crows': 'CDL2CROWS',
    'unique_three_river': 'CDLUNIQUE3RIVER',
    'upside_gap_two_crows': 'CDLUPSIDEGAP2CROWS',
    'xside_gap_three_methods': 'CDLXSIDEGAP3METHODS',
}


def detect_pattern_talib(
    df: pd.DataFrame,
    pattern: str,
    return_strength: bool = False
) -> pd.Series:
    """Detect pattern using TA-Lib.

    Args:
        df: DataFrame with OHLC data
        pattern: Pattern name
        return_strength: Return pattern strength (default: False)

    Returns:
        Series with pattern signals

    Raises:
        ImportError: If TA-Lib not available
        ValueError: If pattern not found
    """
    if not TALIB_AVAILABLE:
        raise ImportError("TA-Lib not available. Install: pip install TA-Lib")

    # Normalize pattern name
    pattern_key = pattern.lower().replace(' ', '_')

    if pattern_key not in TALIB_PATTERNS:
        raise ValueError(
            f"Unknown pattern: {pattern}. "
            f"Available: {', '.join(sorted(TALIB_PATTERNS.keys()))}"
        )

    # Get TA-Lib function name
    talib_func_name = TALIB_PATTERNS[pattern_key]

    # Get function from talib
    if not hasattr(talib, talib_func_name):
        raise ValueError(f"TA-Lib function not found: {talib_func_name}")

    talib_func = getattr(talib, talib_func_name)

    # Extract OHLC
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values

    # Call TA-Lib function
    try:
        result = talib_func(open_prices, high_prices, low_prices, close_prices)
    except Exception as e:
        logger.error(f"TA-Lib pattern detection failed: {e}")
        return pd.Series(0, index=df.index)

    # Convert to series
    result_series = pd.Series(result, index=df.index, name=pattern_key)

    # TA-Lib returns:
    # - 100 for bullish pattern
    # - -100 for bearish pattern
    # - 0 for no pattern

    if return_strength:
        return result_series
    else:
        # Convert to -1, 0, 1
        return (result_series / 100).astype(int)


def detect_all_talib(df: pd.DataFrame) -> pd.DataFrame:
    """Detect all TA-Lib patterns.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with columns for each pattern

    Raises:
        ImportError: If TA-Lib not available
    """
    if not TALIB_AVAILABLE:
        raise ImportError("TA-Lib not available. Install: pip install TA-Lib")

    results = {}

    for pattern_name in sorted(TALIB_PATTERNS.keys()):
        try:
            result = detect_pattern_talib(df, pattern_name, return_strength=False)
            results[pattern_name] = result
        except Exception as e:
            logger.warning(f"Failed to detect pattern {pattern_name}: {e}")
            results[pattern_name] = pd.Series(0, index=df.index)

    return pd.DataFrame(results)


def get_available_patterns_talib() -> List[str]:
    """Get list of available TA-Lib patterns.

    Returns:
        List of pattern names
    """
    return sorted(TALIB_PATTERNS.keys())


def is_talib_available() -> bool:
    """Check if TA-Lib is available.

    Returns:
        True if TA-Lib installed
    """
    return TALIB_AVAILABLE


# Convenience functions for common patterns
def detect_doji(df: pd.DataFrame) -> pd.Series:
    """Detect Doji pattern."""
    return detect_pattern_talib(df, 'doji')


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """Detect Hammer pattern."""
    return detect_pattern_talib(df, 'hammer')


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """Detect Shooting Star pattern."""
    return detect_pattern_talib(df, 'shooting_star')


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detect Engulfing pattern (bullish or bearish)."""
    return detect_pattern_talib(df, 'engulfing')


def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    """Detect Morning Star pattern."""
    return detect_pattern_talib(df, 'morning_star')


def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    """Detect Evening Star pattern."""
    return detect_pattern_talib(df, 'evening_star')


def detect_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    """Detect Three White Soldiers pattern."""
    return detect_pattern_talib(df, 'three_white_soldiers')


def detect_three_black_crows(df: pd.DataFrame) -> pd.Series:
    """Detect Three Black Crows pattern."""
    return detect_pattern_talib(df, 'three_black_crows')
