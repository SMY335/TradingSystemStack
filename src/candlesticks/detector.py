"""
Unified candlestick pattern detection for TradingSystemStack.

Provides a single interface for detecting 60+ candlestick patterns using
TA-Lib (if available) with pandas-ta fallback.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# Pattern categories
BULLISH_PATTERNS = [
    'hammer', 'inverted_hammer', 'bullish_engulfing', 'piercing',
    'morning_star', 'three_white_soldiers', 'bullish_harami'
]

BEARISH_PATTERNS = [
    'hanging_man', 'shooting_star', 'bearish_engulfing', 'dark_cloud_cover',
    'evening_star', 'three_black_crows', 'bearish_harami'
]

REVERSAL_PATTERNS = [
    'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
    'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star',
    'piercing', 'dark_cloud_cover'
]

CONTINUATION_PATTERNS = [
    'rising_three_methods', 'falling_three_methods', 'upside_gap_two_crows',
    'downside_gap_three_methods'
]

INDECISION_PATTERNS = [
    'doji', 'spinning_top', 'high_wave', 'long_legged_doji'
]


class CandlestickDetectorError(Exception):
    """Custom exception for candlestick detection operations."""
    pass


class CandlestickDetector:
    """Unified candlestick pattern detector.

    Detects 60+ candlestick patterns using TA-Lib (preferred) or
    pandas-ta (fallback).

    Examples:
        >>> detector = CandlestickDetector()
        >>> patterns = detector.detect_all(df)
        >>> doji_signals = detector.detect(df, 'doji')
    """

    def __init__(self, use_talib: bool = True):
        """Initialize detector.

        Args:
            use_talib: Prefer TA-Lib if available (default: True)
        """
        self.use_talib = use_talib
        self._talib_available = False
        self._pandas_ta_available = False

        # Check availability
        if use_talib:
            try:
                import talib
                self._talib_available = True
                logger.info("Using TA-Lib for candlestick patterns")
            except ImportError:
                logger.warning("TA-Lib not available, will use pandas fallback")

        # Pandas fallback is always available (pandas is required dependency)
        if not self._talib_available:
            self._pandas_ta_available = True
            logger.info("Using pandas-based pattern detection")

    def detect(
        self,
        df: pd.DataFrame,
        pattern: str,
        return_strength: bool = False
    ) -> pd.Series:
        """Detect a single candlestick pattern.

        Args:
            df: DataFrame with OHLC data
            pattern: Pattern name (e.g., 'doji', 'hammer', 'engulfing')
            return_strength: Return pattern strength instead of binary signal

        Returns:
            Series with pattern signals (1=bullish, -1=bearish, 0=none)
            or pattern strength if return_strength=True

        Examples:
            >>> signals = detector.detect(df, 'doji')
            >>> strength = detector.detect(df, 'hammer', return_strength=True)
        """
        pattern_lower = pattern.lower().replace(' ', '_')

        if self._talib_available:
            from .talib_patterns import detect_pattern_talib
            return detect_pattern_talib(df, pattern_lower, return_strength)

        elif self._pandas_ta_available:
            from .pandas_ta_patterns import detect_pattern_pandas_ta
            return detect_pattern_pandas_ta(df, pattern_lower, return_strength)

        else:
            raise CandlestickDetectorError(
                "No candlestick pattern library available. "
                "Install TA-Lib or pandas-ta: pip install TA-Lib pandas-ta"
            )

    def detect_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Detect all candlestick patterns.

        Args:
            df: DataFrame with OHLC data
            categories: Filter by categories (bullish, bearish, reversal, etc.)

        Returns:
            DataFrame with columns for each pattern (1=bullish, -1=bearish, 0=none)

        Examples:
            >>> all_patterns = detector.detect_all(df)
            >>> bullish_only = detector.detect_all(df, categories=['bullish'])
        """
        if self._talib_available:
            from .talib_patterns import detect_all_talib
            result = detect_all_talib(df)

        elif self._pandas_ta_available:
            from .pandas_ta_patterns import detect_all_pandas_ta
            result = detect_all_pandas_ta(df)

        else:
            raise CandlestickDetectorError(
                "No candlestick pattern library available. "
                "Install TA-Lib or pandas-ta"
            )

        # Filter by categories if specified
        if categories:
            patterns_to_keep = set()
            for category in categories:
                if category.lower() == 'bullish':
                    patterns_to_keep.update(BULLISH_PATTERNS)
                elif category.lower() == 'bearish':
                    patterns_to_keep.update(BEARISH_PATTERNS)
                elif category.lower() == 'reversal':
                    patterns_to_keep.update(REVERSAL_PATTERNS)
                elif category.lower() == 'continuation':
                    patterns_to_keep.update(CONTINUATION_PATTERNS)
                elif category.lower() == 'indecision':
                    patterns_to_keep.update(INDECISION_PATTERNS)

            # Filter columns
            pattern_cols = [col for col in result.columns if any(
                p in col.lower() for p in patterns_to_keep
            )]
            result = result[pattern_cols]

        return result

    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern names.

        Returns:
            List of pattern names

        Examples:
            >>> patterns = detector.get_available_patterns()
            >>> print(patterns[:5])
            ['doji', 'hammer', 'engulfing', 'morning_star', 'evening_star']
        """
        if self._talib_available:
            from .talib_patterns import get_available_patterns_talib
            return get_available_patterns_talib()

        elif self._pandas_ta_available:
            from .pandas_ta_patterns import get_available_patterns_pandas_ta
            return get_available_patterns_pandas_ta()

        return []

    def get_pattern_category(self, pattern: str) -> str:
        """Get category for a pattern.

        Args:
            pattern: Pattern name

        Returns:
            Category name ('bullish', 'bearish', 'reversal', etc.)

        Examples:
            >>> detector.get_pattern_category('hammer')
            'bullish_reversal'
        """
        pattern_lower = pattern.lower()

        categories = []

        if pattern_lower in BULLISH_PATTERNS:
            categories.append('bullish')
        if pattern_lower in BEARISH_PATTERNS:
            categories.append('bearish')
        if pattern_lower in REVERSAL_PATTERNS:
            categories.append('reversal')
        if pattern_lower in CONTINUATION_PATTERNS:
            categories.append('continuation')
        if pattern_lower in INDECISION_PATTERNS:
            categories.append('indecision')

        return '_'.join(categories) if categories else 'unknown'

    def scan_recent(
        self,
        df: pd.DataFrame,
        lookback: int = 10,
        min_strength: int = 50
    ) -> Dict[str, List[int]]:
        """Scan for recent pattern occurrences.

        Args:
            df: DataFrame with OHLC data
            lookback: Number of recent bars to scan
            min_strength: Minimum pattern strength (0-100)

        Returns:
            Dictionary mapping pattern names to list of bar indices

        Examples:
            >>> recent = detector.scan_recent(df, lookback=20)
            >>> print(recent['hammer'])
            [15, 18]
        """
        recent_df = df.tail(lookback)
        patterns = self.detect_all(recent_df)

        results = {}
        for col in patterns.columns:
            # Find non-zero pattern signals
            pattern_indices = patterns[col][patterns[col] != 0].index.tolist()
            if pattern_indices:
                results[col] = pattern_indices

        return results


def detect_single_pattern(
    df: pd.DataFrame,
    pattern: str,
    use_talib: bool = True
) -> pd.Series:
    """Detect a single candlestick pattern (convenience function).

    Args:
        df: DataFrame with OHLC data
        pattern: Pattern name
        use_talib: Prefer TA-Lib if available

    Returns:
        Series with pattern signals

    Examples:
        >>> doji = detect_single_pattern(df, 'doji')
        >>> hammer = detect_single_pattern(df, 'hammer')
    """
    detector = CandlestickDetector(use_talib=use_talib)
    return detector.detect(df, pattern)


def detect_all_patterns(
    df: pd.DataFrame,
    use_talib: bool = True
) -> pd.DataFrame:
    """Detect all candlestick patterns (convenience function).

    Args:
        df: DataFrame with OHLC data
        use_talib: Prefer TA-Lib if available

    Returns:
        DataFrame with all pattern signals

    Examples:
        >>> patterns = detect_all_patterns(df)
    """
    detector = CandlestickDetector(use_talib=use_talib)
    return detector.detect_all(df)
