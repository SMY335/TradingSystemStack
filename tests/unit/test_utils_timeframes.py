"""
Unit tests for utils.timeframes module.
"""
import pytest
import pandas as pd
import numpy as np

from src.utils.timeframes import (
    normalize_timeframe,
    to_pandas_freq,
    to_seconds,
    get_timeframe_name,
    resample_ohlcv,
    get_supported_timeframes,
    is_valid_timeframe,
    compare_timeframes,
    calculate_bars_needed,
    TimeframeError,
)


class TestTimeframeNormalization:
    """Test timeframe normalization."""

    def test_normalize_standard_timeframes(self):
        """Test normalization of standard timeframes."""
        assert normalize_timeframe('1m') == '1m'
        assert normalize_timeframe('1h') == '1h'
        assert normalize_timeframe('1d') == '1d'
        assert normalize_timeframe('1w') == '1w'
        assert normalize_timeframe('1M') == '1M'

    def test_normalize_aliases(self):
        """Test normalization of timeframe aliases."""
        assert normalize_timeframe('1min') == '1m'
        assert normalize_timeframe('5min') == '5m'
        assert normalize_timeframe('1hour') == '1h'
        assert normalize_timeframe('1day') == '1d'
        assert normalize_timeframe('1week') == '1w'
        assert normalize_timeframe('1month') == '1M'

    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        assert normalize_timeframe('1H') == '1h'
        assert normalize_timeframe('1D') == '1d'
        assert normalize_timeframe('1W') == '1w'

    def test_normalize_invalid(self):
        """Test normalization of invalid timeframe."""
        with pytest.raises(TimeframeError, match="Unsupported timeframe"):
            normalize_timeframe('invalid')

        with pytest.raises(TimeframeError):
            normalize_timeframe('99x')


class TestTimeframeConversions:
    """Test timeframe conversions."""

    def test_to_pandas_freq(self):
        """Test conversion to pandas frequency."""
        assert to_pandas_freq('1m') == '1min'
        assert to_pandas_freq('1h') == '1H'
        assert to_pandas_freq('1d') == '1D'
        assert to_pandas_freq('1w') == '1W'
        assert to_pandas_freq('1M') == '1M'

    def test_to_seconds(self):
        """Test conversion to seconds."""
        assert to_seconds('1m') == 60
        assert to_seconds('5m') == 300
        assert to_seconds('1h') == 3600
        assert to_seconds('4h') == 14400
        assert to_seconds('1d') == 86400
        assert to_seconds('1w') == 604800

    def test_get_timeframe_name(self):
        """Test getting human-readable names."""
        assert get_timeframe_name('1m') == '1 Minute'
        assert get_timeframe_name('1h') == '1 Hour'
        assert get_timeframe_name('1d') == '1 Day'
        assert get_timeframe_name('1w') == '1 Week'


class TestTimeframeResampling:
    """Test OHLCV resampling."""

    def test_resample_1m_to_1h(self):
        """Test resampling 1-minute data to 1-hour."""
        # Create 1-minute data for 1 hour (60 bars)
        dates = pd.date_range('2024-01-01', periods=60, freq='1min')
        df = pd.DataFrame({
            'open': range(100, 160),
            'high': range(101, 161),
            'low': range(99, 159),
            'close': range(100, 160),
            'volume': [100] * 60
        }, index=dates)

        # Resample to 1h
        resampled = resample_ohlcv(df, '1h')

        assert len(resampled) == 1
        assert resampled['open'].iloc[0] == 100  # First open
        assert resampled['high'].iloc[0] == 160  # Max high
        assert resampled['low'].iloc[0] == 99    # Min low
        assert resampled['close'].iloc[0] == 159 # Last close
        assert resampled['volume'].iloc[0] == 6000  # Sum

    def test_resample_1h_to_1d(self):
        """Test resampling 1-hour data to daily."""
        dates = pd.date_range('2024-01-01', periods=24, freq='1H')
        df = pd.DataFrame({
            'open': [100] * 24,
            'high': [105] * 24,
            'low': [95] * 24,
            'close': [102] * 24,
            'volume': [1000] * 24
        }, index=dates)

        resampled = resample_ohlcv(df, '1d')

        assert len(resampled) == 1
        assert resampled['open'].iloc[0] == 100
        assert resampled['high'].iloc[0] == 105
        assert resampled['low'].iloc[0] == 95
        assert resampled['close'].iloc[0] == 102
        assert resampled['volume'].iloc[0] == 24000

    def test_resample_no_datetime_index(self):
        """Test resampling with non-datetime index."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        with pytest.raises(ValueError, match="DatetimeIndex"):
            resample_ohlcv(df, '1h')

    def test_resample_smaller_timeframe_error(self):
        """Test error when resampling to smaller timeframe."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        }, index=dates)

        with pytest.raises(TimeframeError, match="Cannot resample to smaller timeframe"):
            resample_ohlcv(df, '1m', source_tf='1h')


class TestTimeframeUtilities:
    """Test utility functions."""

    def test_get_supported_timeframes(self):
        """Test getting list of supported timeframes."""
        tfs = get_supported_timeframes()

        assert isinstance(tfs, list)
        assert '1m' in tfs
        assert '1h' in tfs
        assert '1d' in tfs
        assert '1w' in tfs
        assert '1M' in tfs

        # Should be sorted by duration
        assert tfs.index('1m') < tfs.index('1h')
        assert tfs.index('1h') < tfs.index('1d')
        assert tfs.index('1d') < tfs.index('1w')

    def test_is_valid_timeframe(self):
        """Test timeframe validation."""
        assert is_valid_timeframe('1m') is True
        assert is_valid_timeframe('1h') is True
        assert is_valid_timeframe('1d') is True
        assert is_valid_timeframe('1hour') is True  # Alias

        assert is_valid_timeframe('invalid') is False
        assert is_valid_timeframe('99x') is False

    def test_compare_timeframes(self):
        """Test comparing timeframes."""
        assert compare_timeframes('1m', '1h') == -1  # 1m < 1h
        assert compare_timeframes('1h', '1m') == 1   # 1h > 1m
        assert compare_timeframes('1h', '1h') == 0   # Equal

        assert compare_timeframes('1h', '1d') == -1
        assert compare_timeframes('1d', '1h') == 1

    def test_calculate_bars_needed(self):
        """Test calculating bars needed for conversion."""
        # 1 hour from 1-minute bars
        assert calculate_bars_needed('1h', '1m', 1) == 60

        # 1 day from 1-hour bars
        assert calculate_bars_needed('1d', '1h', 1) == 24

        # 1 week from 1-day bars
        assert calculate_bars_needed('1w', '1d', 1) == 7

        # Multiple target bars
        assert calculate_bars_needed('1h', '1m', 5) == 300

    def test_calculate_bars_invalid(self):
        """Test error when target < source."""
        with pytest.raises(TimeframeError, match="must be >= source"):
            calculate_bars_needed('1m', '1h', 1)
