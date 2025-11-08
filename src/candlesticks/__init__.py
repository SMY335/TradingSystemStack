"""
Candlestick pattern recognition for TradingSystemStack.

Provides detection of 60+ candlestick patterns using TA-Lib (preferred)
or pandas-based fallback implementation.

Examples:
    >>> from src.candlesticks import CandlestickDetector, detect_single_pattern
    >>>
    >>> # Using detector class
    >>> detector = CandlestickDetector()
    >>> patterns = detector.detect_all(df)
    >>> doji_signals = detector.detect(df, 'doji')
    >>>
    >>> # Using convenience functions
    >>> hammer = detect_single_pattern(df, 'hammer')
    >>> all_patterns = detect_all_patterns(df)
"""

from .detector import (
    CandlestickDetector,
    CandlestickDetectorError,
    detect_single_pattern,
    detect_all_patterns,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS,
    REVERSAL_PATTERNS,
    CONTINUATION_PATTERNS,
    INDECISION_PATTERNS,
)

# Try to expose TA-Lib functions if available
try:
    from .talib_patterns import (
        detect_doji,
        detect_hammer,
        detect_shooting_star,
        detect_engulfing,
        detect_morning_star,
        detect_evening_star,
        detect_three_white_soldiers,
        detect_three_black_crows,
        is_talib_available,
    )
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


__all__ = [
    # Main detector
    'CandlestickDetector',
    'CandlestickDetectorError',

    # Convenience functions
    'detect_single_pattern',
    'detect_all_patterns',

    # Pattern categories
    'BULLISH_PATTERNS',
    'BEARISH_PATTERNS',
    'REVERSAL_PATTERNS',
    'CONTINUATION_PATTERNS',
    'INDECISION_PATTERNS',

    # Availability flag
    'TALIB_AVAILABLE',
]

# Add TA-Lib specific functions if available
if TALIB_AVAILABLE:
    __all__.extend([
        'detect_doji',
        'detect_hammer',
        'detect_shooting_star',
        'detect_engulfing',
        'detect_morning_star',
        'detect_evening_star',
        'detect_three_white_soldiers',
        'detect_three_black_crows',
        'is_talib_available',
    ])
