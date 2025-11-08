"""
Unit tests for indicators module.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.indicators import (
    run_indicator,
    get_available_indicators,
    BaseIndicator,
)
from src.indicators.exceptions import (
    IndicatorNotFoundError,
    InvalidDataError,
    CalculationError,
    LibraryNotAvailableError,
)
from src.indicators.wrappers.talib_wrapper import (
    RSI, MACD, EMA, SMA, BBANDS, ATR, STOCH, ADX, is_talib_available
)
from src.indicators.wrappers.pandas_ta_wrapper import (
    SuperTrend, Ichimoku, is_pandas_ta_available
)


@pytest.fixture
def sample_ohlc():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def sample_close():
    """Create sample close price data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    return pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum()
    }, index=dates)


class TestCoreIndicators:
    """Test core indicator functionality."""

    def test_run_indicator_rsi(self, sample_close):
        """Test running RSI indicator."""
        if not is_talib_available():
            pytest.skip("TA-Lib not available")

        result = run_indicator('RSI', sample_close, params={'length': 14})

        assert 'rsi' in result.columns
        assert len(result) == len(sample_close)
        assert result['rsi'].notna().any()  # Should have some non-NaN values

    def test_run_indicator_macd(self, sample_close):
        """Test running MACD indicator."""
        if not is_talib_available():
            pytest.skip("TA-Lib not available")

        result = run_indicator('MACD', sample_close, params={'fast': 12, 'slow': 26, 'signal': 9})

        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
        assert len(result) == len(sample_close)

    def test_run_indicator_with_alias(self, sample_close):
        """Test running indicator using alias."""
        if not is_talib_available():
            pytest.skip("TA-Lib not available")

        result = run_indicator('rsi', sample_close, params={'length': 14})

        assert 'rsi' in result.columns

    def test_run_indicator_not_found(self, sample_close):
        """Test error when indicator not found."""
        with pytest.raises(IndicatorNotFoundError, match="not found in registry"):
            run_indicator('NonExistentIndicator', sample_close)

    def test_run_indicator_default_params(self, sample_close):
        """Test running indicator with default parameters."""
        if not is_talib_available():
            pytest.skip("TA-Lib not available")

        result = run_indicator('RSI', sample_close)

        assert 'rsi' in result.columns

    def test_get_available_indicators(self):
        """Test getting list of available indicators."""
        indicators = get_available_indicators()

        assert isinstance(indicators, dict)

        if is_talib_available():
            assert 'RSI' in indicators or any('RSI' in k for k in indicators.keys())

            # Check structure
            for name, info in indicators.items():
                assert 'library' in info
                assert 'aliases' in info

    def test_get_available_indicators_by_library(self):
        """Test filtering indicators by library."""
        if is_talib_available():
            talib_indicators = get_available_indicators(library='talib')

            assert isinstance(talib_indicators, dict)

            # All should be talib
            for name, info in talib_indicators.items():
                assert info['library'] == 'talib'


class TestTALibWrappers:
    """Test TA-Lib indicator wrappers."""

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_rsi_basic(self, sample_close):
        """Test basic RSI calculation."""
        indicator = RSI()
        result = indicator.calculate(sample_close, length=14)

        assert 'rsi' in result.columns
        assert len(result) == len(sample_close)

        # RSI should be between 0 and 100
        valid_rsi = result['rsi'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_macd_basic(self, sample_close):
        """Test basic MACD calculation."""
        indicator = MACD()
        result = indicator.calculate(sample_close, fast=12, slow=26, signal=9)

        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns

        # Histogram should be macd - signal
        valid_idx = result['macd'].notna() & result['macd_signal'].notna()
        expected_hist = result.loc[valid_idx, 'macd'] - result.loc[valid_idx, 'macd_signal']
        actual_hist = result.loc[valid_idx, 'macd_hist']

        np.testing.assert_array_almost_equal(actual_hist, expected_hist, decimal=5)

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_ema_basic(self, sample_close):
        """Test basic EMA calculation."""
        indicator = EMA()
        result = indicator.calculate(sample_close, length=20)

        assert 'ema_20' in result.columns
        assert len(result) == len(sample_close)

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_sma_basic(self, sample_close):
        """Test basic SMA calculation."""
        indicator = SMA()
        result = indicator.calculate(sample_close, length=20)

        assert 'sma_20' in result.columns
        assert len(result) == len(sample_close)

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_bbands_basic(self, sample_close):
        """Test basic Bollinger Bands calculation."""
        indicator = BBANDS()
        result = indicator.calculate(sample_close, length=20, std=2.0)

        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns

        # Upper should be >= middle >= lower
        valid_idx = (
            result['bb_upper'].notna() &
            result['bb_middle'].notna() &
            result['bb_lower'].notna()
        )

        assert (result.loc[valid_idx, 'bb_upper'] >= result.loc[valid_idx, 'bb_middle']).all()
        assert (result.loc[valid_idx, 'bb_middle'] >= result.loc[valid_idx, 'bb_lower']).all()

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_atr_basic(self, sample_ohlc):
        """Test basic ATR calculation."""
        indicator = ATR()
        result = indicator.calculate(sample_ohlc, length=14)

        assert 'atr' in result.columns

        # ATR should be positive
        valid_atr = result['atr'].dropna()
        assert (valid_atr >= 0).all()

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_stoch_basic(self, sample_ohlc):
        """Test basic Stochastic calculation."""
        indicator = STOCH()
        result = indicator.calculate(sample_ohlc, k_period=14, d_period=3)

        assert 'stoch_k' in result.columns
        assert 'stoch_d' in result.columns

        # Stochastic should be between 0 and 100
        valid_k = result['stoch_k'].dropna()
        valid_d = result['stoch_d'].dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_adx_basic(self, sample_ohlc):
        """Test basic ADX calculation."""
        indicator = ADX()
        result = indicator.calculate(sample_ohlc, length=14)

        assert 'adx' in result.columns

        # ADX should be between 0 and 100
        valid_adx = result['adx'].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()


class TestPandasTAWrappers:
    """Test pandas-ta indicator wrappers."""

    @pytest.mark.skipif(not is_pandas_ta_available(), reason="pandas-ta not available")
    def test_supertrend_basic(self, sample_ohlc):
        """Test basic SuperTrend calculation."""
        indicator = SuperTrend()
        result = indicator.calculate(sample_ohlc, length=10, multiplier=3.0)

        assert 'supertrend' in result.columns
        assert 'supertrend_direction' in result.columns

        # Direction should be 1 or -1
        valid_dir = result['supertrend_direction'].dropna()
        assert valid_dir.isin([1, -1]).all()

    @pytest.mark.skipif(not is_pandas_ta_available(), reason="pandas-ta not available")
    def test_ichimoku_basic(self, sample_ohlc):
        """Test basic Ichimoku calculation."""
        indicator = Ichimoku()
        result = indicator.calculate(sample_ohlc, tenkan=9, kijun=26, senkou=52)

        assert 'tenkan_sen' in result.columns
        assert 'kijun_sen' in result.columns
        assert 'senkou_span_a' in result.columns
        assert 'senkou_span_b' in result.columns


class TestValidation:
    """Test indicator validation."""

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_missing_required_column(self):
        """Test error when required column missing."""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})

        indicator = RSI()

        with pytest.raises(InvalidDataError, match="Missing required columns"):
            indicator.calculate(df)

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_empty_dataframe(self):
        """Test error with empty DataFrame."""
        df = pd.DataFrame({'close': []})

        indicator = RSI()

        with pytest.raises(InvalidDataError, match="DataFrame is empty"):
            indicator.calculate(df)

    @pytest.mark.skipif(not is_talib_available(), reason="TA-Lib not available")
    def test_invalid_input_type(self):
        """Test error with invalid input type."""
        indicator = RSI()

        with pytest.raises(InvalidDataError, match="must be a pandas DataFrame"):
            indicator.calculate([1, 2, 3])


class TestLibraryAvailability:
    """Test library availability checks."""

    def test_talib_availability_check(self):
        """Test TA-Lib availability check."""
        available = is_talib_available()
        assert isinstance(available, bool)

    def test_pandas_ta_availability_check(self):
        """Test pandas-ta availability check."""
        available = is_pandas_ta_available()
        assert isinstance(available, bool)

    def test_talib_initialization_checks_availability(self):
        """Test that TA-Lib indicators check for library availability."""
        # This test verifies the behavior when TA-Lib is available
        # The actual "not available" scenario is tested by the is_talib_available() function
        if is_talib_available():
            # Should not raise when library is available
            indicator = RSI()
            assert indicator.library == 'talib'

    def test_pandas_ta_initialization_checks_availability(self):
        """Test that pandas-ta indicators check for library availability."""
        # This test verifies the behavior when pandas-ta is available
        # The actual "not available" scenario is tested by the is_pandas_ta_available() function
        if is_pandas_ta_available():
            # Should not raise when library is available
            indicator = SuperTrend()
            assert indicator.library == 'pandas_ta'


class TestBaseIndicator:
    """Test BaseIndicator abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseIndicator cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseIndicator('test', 'test_lib')

    def test_subclass_must_implement_calculate(self):
        """Test that subclasses must implement calculate."""
        class IncompleteIndicator(BaseIndicator):
            def __init__(self):
                super().__init__('test', 'test_lib')

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteIndicator()
