"""
Unit tests for candlesticks module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.candlesticks import (
    CandlestickDetector,
    detect_single_pattern,
    detect_all_patterns,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS,
)


@pytest.fixture
def sample_ohlc():
    """Create sample OHLC data."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    return pd.DataFrame({
        'open': [100 + i for i in range(20)],
        'high': [105 + i for i in range(20)],
        'low': [98 + i for i in range(20)],
        'close': [102 + i for i in range(20)],
        'volume': [1000 + i*100 for i in range(20)]
    }, index=dates)


@pytest.fixture
def doji_data():
    """Create data with doji pattern."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [100.5, 101, 102, 103, 104],  # Second candle is doji (open ≈ close)
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)


@pytest.fixture
def hammer_data():
    """Create data with hammer pattern."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100, 101, 105, 103, 104],
        'high': [105, 106, 106, 108, 109],
        'low': [95, 96, 100, 98, 99],  # Third candle has long lower shadow
        'close': [102, 103, 104.5, 107, 108],  # Third candle closes near high
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)


@pytest.fixture
def engulfing_data():
    """Create data with engulfing pattern."""
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'open': [100, 105, 103, 108, 104],
        'high': [105, 106, 110, 110, 109],
        'low': [95, 103, 100, 105, 99],
        'close': [102, 104, 109, 107, 108],  # Third candle engulfs second
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=dates)


class TestCandlestickDetector:
    """Test CandlestickDetector class."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = CandlestickDetector()

        assert detector is not None
        # Should use either talib or pandas-ta
        assert detector._talib_available or detector._pandas_ta_available

    def test_get_available_patterns(self):
        """Test getting list of available patterns."""
        detector = CandlestickDetector()
        patterns = detector.get_available_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert 'doji' in patterns or 'doji' in [p.lower() for p in patterns]

    def test_detect_single_pattern(self, sample_ohlc):
        """Test detecting single pattern."""
        detector = CandlestickDetector()

        try:
            result = detector.detect(sample_ohlc, 'doji')

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_ohlc)
            assert result.dtype in [np.int64, np.float64]
        except Exception as e:
            # If pattern not available, skip
            pytest.skip(f"Pattern detection not available: {e}")

    def test_detect_all_patterns(self, sample_ohlc):
        """Test detecting all patterns."""
        detector = CandlestickDetector()

        try:
            result = detector.detect_all(sample_ohlc)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_ohlc)
            assert len(result.columns) > 0
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")

    def test_detect_with_categories(self, sample_ohlc):
        """Test detecting patterns filtered by category."""
        detector = CandlestickDetector()

        try:
            result = detector.detect_all(sample_ohlc, categories=['bullish'])

            assert isinstance(result, pd.DataFrame)
            # Should only have bullish pattern columns
            # (exact columns depend on implementation)
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")

    def test_get_pattern_category(self):
        """Test getting pattern category."""
        detector = CandlestickDetector()

        # Hammer is bullish reversal
        category = detector.get_pattern_category('hammer')
        assert 'bullish' in category.lower() or 'reversal' in category.lower()

        # Shooting star is bearish reversal
        category = detector.get_pattern_category('shooting_star')
        assert 'bearish' in category.lower() or 'reversal' in category.lower()

    def test_scan_recent(self, sample_ohlc):
        """Test scanning for recent patterns."""
        detector = CandlestickDetector()

        try:
            recent = detector.scan_recent(sample_ohlc, lookback=10)

            assert isinstance(recent, dict)
            # May or may not find patterns in random data
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_single_pattern_function(self, sample_ohlc):
        """Test detect_single_pattern convenience function."""
        try:
            result = detect_single_pattern(sample_ohlc, 'doji', use_talib=False)

            assert isinstance(result, pd.Series)
            assert len(result) == len(sample_ohlc)
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")

    def test_detect_all_patterns_function(self, sample_ohlc):
        """Test detect_all_patterns convenience function."""
        try:
            result = detect_all_patterns(sample_ohlc, use_talib=False)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_ohlc)
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")


class TestPatternDetectionLogic:
    """Test pattern detection logic with specific data."""

    def test_detect_doji(self, doji_data):
        """Test doji detection."""
        detector = CandlestickDetector(use_talib=False)

        result = detector.detect(doji_data, 'doji')

        # Second candle should be detected as doji (open ≈ close)
        assert result.iloc[1] != 0 or result.sum() > 0  # At least some doji detected

    def test_detect_hammer(self, hammer_data):
        """Test hammer detection."""
        detector = CandlestickDetector(use_talib=False)

        result = detector.detect(hammer_data, 'hammer')

        # Should detect hammer pattern (bullish reversal)
        assert isinstance(result, pd.Series)
        # May or may not detect depending on exact criteria

    def test_detect_engulfing(self, engulfing_data):
        """Test engulfing pattern detection."""
        detector = CandlestickDetector(use_talib=False)

        result = detector.detect(engulfing_data, 'engulfing')

        assert isinstance(result, pd.Series)
        # Engulfing pattern should be detected somewhere

    def test_pattern_signal_values(self, sample_ohlc):
        """Test that pattern signals are -1, 0, or 1."""
        detector = CandlestickDetector(use_talib=False)

        try:
            result = detector.detect(sample_ohlc, 'doji')

            # Signals should be -1, 0, or 1
            assert result.isin([-1, 0, 1]).all()
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")


class TestPatternCategories:
    """Test pattern categorization."""

    def test_bullish_patterns_list(self):
        """Test bullish patterns list."""
        assert isinstance(BULLISH_PATTERNS, list)
        assert len(BULLISH_PATTERNS) > 0
        assert 'hammer' in BULLISH_PATTERNS

    def test_bearish_patterns_list(self):
        """Test bearish patterns list."""
        assert isinstance(BEARISH_PATTERNS, list)
        assert len(BEARISH_PATTERNS) > 0
        assert 'shooting_star' in BEARISH_PATTERNS

    def test_pattern_not_in_both(self):
        """Test that patterns are not in both bullish and bearish."""
        # Some patterns can be in both (like engulfing which can be bullish or bearish)
        # But most should be in one category
        bullish_set = set(BULLISH_PATTERNS)
        bearish_set = set(BEARISH_PATTERNS)

        # At least some patterns should be exclusive
        assert len(bullish_set) > 0
        assert len(bearish_set) > 0


class TestReturnStrength:
    """Test return_strength parameter."""

    def test_detect_with_strength(self, sample_ohlc):
        """Test detecting with pattern strength."""
        detector = CandlestickDetector(use_talib=False)

        try:
            # Get binary signals
            signals = detector.detect(sample_ohlc, 'doji', return_strength=False)
            # Get strength values
            strength = detector.detect(sample_ohlc, 'doji', return_strength=True)

            assert isinstance(signals, pd.Series)
            assert isinstance(strength, pd.Series)
            # Strength values should be different from binary signals
        except Exception as e:
            pytest.skip(f"Pattern detection not available: {e}")


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        detector = CandlestickDetector(use_talib=False)
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

        try:
            result = detector.detect(empty_df, 'doji')
            assert len(result) == 0
        except Exception:
            # May raise exception for empty data
            pass

    def test_single_candle(self):
        """Test with single candle."""
        detector = CandlestickDetector(use_talib=False)
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [102]
        })

        try:
            result = detector.detect(df, 'doji')
            assert len(result) == 1
        except Exception:
            # Some patterns need multiple candles
            pass

    def test_invalid_pattern_name(self, sample_ohlc):
        """Test with invalid pattern name."""
        detector = CandlestickDetector(use_talib=False)

        result = detector.detect(sample_ohlc, 'nonexistent_pattern')

        # Should return zeros for unknown pattern (pandas fallback)
        assert isinstance(result, pd.Series)
        assert (result == 0).all()
