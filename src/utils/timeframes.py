"""
Timeframe conversion utilities for TradingSystemStack.

Handles conversion between different timeframe formats and validation.
"""
from typing import Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TimeframeError(Exception):
    """Custom exception for timeframe operations."""
    pass


# Standard timeframe mappings
TIMEFRAME_MAP = {
    # Minutes
    '1m': {'pandas': '1min', 'seconds': 60, 'name': '1 Minute'},
    '3m': {'pandas': '3min', 'seconds': 180, 'name': '3 Minutes'},
    '5m': {'pandas': '5min', 'seconds': 300, 'name': '5 Minutes'},
    '15m': {'pandas': '15min', 'seconds': 900, 'name': '15 Minutes'},
    '30m': {'pandas': '30min', 'seconds': 1800, 'name': '30 Minutes'},
    '45m': {'pandas': '45min', 'seconds': 2700, 'name': '45 Minutes'},

    # Hours
    '1h': {'pandas': '1H', 'seconds': 3600, 'name': '1 Hour'},
    '2h': {'pandas': '2H', 'seconds': 7200, 'name': '2 Hours'},
    '3h': {'pandas': '3H', 'seconds': 10800, 'name': '3 Hours'},
    '4h': {'pandas': '4H', 'seconds': 14400, 'name': '4 Hours'},
    '6h': {'pandas': '6H', 'seconds': 21600, 'name': '6 Hours'},
    '8h': {'pandas': '8H', 'seconds': 28800, 'name': '8 Hours'},
    '12h': {'pandas': '12H', 'seconds': 43200, 'name': '12 Hours'},

    # Days
    '1d': {'pandas': '1D', 'seconds': 86400, 'name': '1 Day'},
    '3d': {'pandas': '3D', 'seconds': 259200, 'name': '3 Days'},

    # Weeks
    '1w': {'pandas': '1W', 'seconds': 604800, 'name': '1 Week'},

    # Months
    '1M': {'pandas': '1M', 'seconds': 2592000, 'name': '1 Month'},  # Approximate
}


# Alias mappings (alternative names for same timeframe)
TIMEFRAME_ALIASES = {
    '1min': '1m',
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1hour': '1h',
    '4hour': '4h',
    '1day': '1d',
    '1week': '1w',
    '1month': '1M',

    # CCXT style
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',

    # Numeric only
    '1': '1m',
    '5': '5m',
    '15': '15m',
    '60': '1h',
    '240': '4h',
    'D': '1d',
    'W': '1w',
    'M': '1M',
}


def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe string to standard format.

    Args:
        tf: Timeframe string in any supported format

    Returns:
        Normalized timeframe (e.g., '1h', '1d', '1w')

    Raises:
        TimeframeError: If timeframe format is invalid

    Examples:
        >>> normalize_timeframe('1hour')
        '1h'
        >>> normalize_timeframe('1day')
        '1d'
        >>> normalize_timeframe('5min')
        '5m'
    """
    tf_lower = tf.lower().strip()

    # Check if already normalized
    if tf_lower in TIMEFRAME_MAP:
        return tf_lower

    # Check aliases
    if tf_lower in TIMEFRAME_ALIASES:
        normalized = TIMEFRAME_ALIASES[tf_lower]
        logger.debug(f"Normalized {tf} → {normalized}")
        return normalized

    raise TimeframeError(
        f"Unsupported timeframe: {tf}. "
        f"Supported: {', '.join(sorted(TIMEFRAME_MAP.keys()))}"
    )


def to_pandas_freq(tf: str) -> str:
    """Convert timeframe to pandas frequency string.

    Args:
        tf: Timeframe string

    Returns:
        Pandas frequency string (e.g., '1H', '1D')

    Examples:
        >>> to_pandas_freq('1h')
        '1H'
        >>> to_pandas_freq('1d')
        '1D'
    """
    tf_norm = normalize_timeframe(tf)
    return TIMEFRAME_MAP[tf_norm]['pandas']


def to_seconds(tf: str) -> int:
    """Convert timeframe to seconds.

    Args:
        tf: Timeframe string

    Returns:
        Number of seconds in timeframe

    Examples:
        >>> to_seconds('1m')
        60
        >>> to_seconds('1h')
        3600
        >>> to_seconds('1d')
        86400
    """
    tf_norm = normalize_timeframe(tf)
    return TIMEFRAME_MAP[tf_norm]['seconds']


def get_timeframe_name(tf: str) -> str:
    """Get human-readable timeframe name.

    Args:
        tf: Timeframe string

    Returns:
        Human-readable name

    Examples:
        >>> get_timeframe_name('1h')
        '1 Hour'
        >>> get_timeframe_name('1d')
        '1 Day'
    """
    tf_norm = normalize_timeframe(tf)
    return TIMEFRAME_MAP[tf_norm]['name']


def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str,
    source_tf: Optional[str] = None
) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe.

    Args:
        df: DataFrame with OHLCV data (index must be DatetimeIndex)
        target_tf: Target timeframe
        source_tf: Source timeframe (for validation, optional)

    Returns:
        Resampled DataFrame

    Raises:
        TimeframeError: If resampling fails or timeframes invalid
        ValueError: If DataFrame doesn't have DatetimeIndex

    Examples:
        >>> df_1h = resample_ohlcv(df_1m, '1h')  # 1m → 1h
        >>> df_1d = resample_ohlcv(df_1h, '1d')  # 1h → 1d
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for resampling")

    target_norm = normalize_timeframe(target_tf)
    target_freq = to_pandas_freq(target_norm)

    logger.debug(f"Resampling to {target_norm} (pandas: {target_freq})")

    try:
        # Validate source timeframe if provided
        if source_tf:
            source_norm = normalize_timeframe(source_tf)
            source_seconds = to_seconds(source_norm)
            target_seconds = to_seconds(target_norm)

            if target_seconds < source_seconds:
                raise TimeframeError(
                    f"Cannot resample to smaller timeframe: {source_norm} → {target_norm}"
                )

        # OHLCV resampling rules
        resampled = df.resample(target_freq).agg({
            'open': 'first',   # First open
            'high': 'max',     # Highest high
            'low': 'min',      # Lowest low
            'close': 'last',   # Last close
            'volume': 'sum'    # Sum of volume
        })

        # Drop rows with NaN (incomplete periods)
        resampled = resampled.dropna()

        logger.debug(f"Resampled {len(df)} → {len(resampled)} bars")

        return resampled

    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        raise TimeframeError(f"Failed to resample: {e}") from e


def get_supported_timeframes() -> list[str]:
    """Get list of all supported timeframes.

    Returns:
        Sorted list of timeframe strings

    Examples:
        >>> tfs = get_supported_timeframes()
        >>> print(tfs)
        ['1m', '3m', '5m', '15m', ...]
    """
    return sorted(TIMEFRAME_MAP.keys(), key=lambda x: to_seconds(x))


def is_valid_timeframe(tf: str) -> bool:
    """Check if timeframe string is valid.

    Args:
        tf: Timeframe string to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> is_valid_timeframe('1h')
        True
        >>> is_valid_timeframe('invalid')
        False
    """
    try:
        normalize_timeframe(tf)
        return True
    except TimeframeError:
        return False


def compare_timeframes(tf1: str, tf2: str) -> int:
    """Compare two timeframes.

    Args:
        tf1: First timeframe
        tf2: Second timeframe

    Returns:
        -1 if tf1 < tf2, 0 if equal, 1 if tf1 > tf2

    Examples:
        >>> compare_timeframes('1m', '1h')
        -1
        >>> compare_timeframes('1d', '1h')
        1
        >>> compare_timeframes('1h', '1h')
        0
    """
    seconds1 = to_seconds(tf1)
    seconds2 = to_seconds(tf2)

    if seconds1 < seconds2:
        return -1
    elif seconds1 > seconds2:
        return 1
    else:
        return 0


def calculate_bars_needed(
    target_tf: str,
    source_tf: str,
    target_bars: int = 1
) -> int:
    """Calculate how many source bars needed for target bars.

    Args:
        target_tf: Target timeframe
        source_tf: Source timeframe
        target_bars: Number of target bars needed

    Returns:
        Number of source bars needed

    Examples:
        >>> calculate_bars_needed('1h', '1m', 1)  # 1 hour from 1min bars
        60
        >>> calculate_bars_needed('1d', '1h', 1)  # 1 day from 1h bars
        24
    """
    target_seconds = to_seconds(target_tf)
    source_seconds = to_seconds(source_tf)

    if target_seconds < source_seconds:
        raise TimeframeError(f"Target timeframe must be >= source timeframe")

    bars_per_target = target_seconds // source_seconds
    total_bars = bars_per_target * target_bars

    return total_bars
