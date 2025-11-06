"""
Unit tests for SuperTrendStrategy validation
Tests all parameter validation for the SuperTrend strategy
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.supertrend_strategy import SuperTrendStrategy


class TestSuperTrendValidation:
    """Test SuperTrendStrategy parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        strategy = SuperTrendStrategy(period=10, multiplier=3.0)
        assert strategy.params['period'] == 10
        assert strategy.params['multiplier'] == 3.0

    def test_default_parameters(self):
        """Test default parameter values"""
        strategy = SuperTrendStrategy()
        assert strategy.params['period'] == 10
        assert strategy.params['multiplier'] == 3.0

    def test_period_not_int_rejected(self):
        """Test that non-integer period is rejected"""
        with pytest.raises(TypeError, match="period must be int"):
            SuperTrendStrategy(period=10.5)

    def test_period_too_small_rejected(self):
        """Test that period < 1 is rejected"""
        with pytest.raises(ValueError, match="period must be >= 1"):
            SuperTrendStrategy(period=0)

    def test_period_too_large_rejected(self):
        """Test that period > 50 is rejected"""
        with pytest.raises(ValueError, match="period cannot exceed 50"):
            SuperTrendStrategy(period=51)

    def test_multiplier_not_numeric_rejected(self):
        """Test that non-numeric multiplier is rejected"""
        with pytest.raises(TypeError, match="multiplier must be numeric"):
            SuperTrendStrategy(multiplier="3.0")

    def test_multiplier_non_positive_rejected(self):
        """Test that multiplier <= 0 is rejected"""
        with pytest.raises(ValueError, match="multiplier must be positive"):
            SuperTrendStrategy(multiplier=0)

        with pytest.raises(ValueError, match="multiplier must be positive"):
            SuperTrendStrategy(multiplier=-1.0)

    def test_multiplier_too_large_rejected(self):
        """Test that multiplier > 10 is rejected"""
        with pytest.raises(ValueError, match="multiplier cannot exceed 10"):
            SuperTrendStrategy(multiplier=10.5)

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values
        strategy = SuperTrendStrategy(period=1, multiplier=0.1)
        assert strategy.params['period'] == 1
        assert strategy.params['multiplier'] == 0.1

        # Maximum valid values
        strategy = SuperTrendStrategy(period=50, multiplier=10.0)
        assert strategy.params['period'] == 50
        assert strategy.params['multiplier'] == 10.0

    def test_param_schema(self):
        """Test that parameter schema is correctly defined"""
        strategy = SuperTrendStrategy()
        schema = strategy.get_param_schema()

        assert 'period' in schema
        assert schema['period']['type'] == 'int'
        assert schema['period']['min'] == 7
        assert schema['period']['max'] == 20
        assert schema['period']['default'] == 10

        assert 'multiplier' in schema
        assert schema['multiplier']['type'] == 'float'
        assert schema['multiplier']['min'] == 1.0
        assert schema['multiplier']['max'] == 5.0


class TestSuperTrendSignals:
    """Test SuperTrendStrategy signal generation and calculations"""

    def test_generate_signals_with_valid_data(self):
        """Test signal generation with valid OHLC data"""
        strategy = SuperTrendStrategy(period=10, multiplier=3.0)

        # Create sample OHLC data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)),
            'low': close - np.abs(np.random.randn(100))
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)
        assert len(exits) == len(df)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_calculate_atr(self):
        """Test ATR calculation"""
        strategy = SuperTrendStrategy(period=10)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)),
            'low': close - np.abs(np.random.randn(100))
        }, index=dates)

        atr = strategy.calculate_atr(df, period=10)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)
        # ATR should be positive (or NaN for initial values)
        valid_atr = atr.dropna()
        if len(valid_atr) > 0:
            assert (valid_atr >= 0).all()

    def test_calculate_supertrend(self):
        """Test SuperTrend calculation"""
        strategy = SuperTrendStrategy(period=10, multiplier=3.0)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)),
            'low': close - np.abs(np.random.randn(100))
        }, index=dates)

        supertrend, trend = strategy.calculate_supertrend(df)

        assert isinstance(supertrend, pd.Series)
        assert isinstance(trend, pd.Series)
        assert len(supertrend) == len(df)
        assert len(trend) == len(df)

        # Trend should be 1 (UP) or -1 (DOWN)
        valid_trend = trend.dropna()
        if len(valid_trend) > 0:
            assert set(valid_trend.unique()).issubset({-1, 1})

    def test_get_supertrend_for_plot(self):
        """Test SuperTrend data for plotting"""
        strategy = SuperTrendStrategy(period=10, multiplier=3.0)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)),
            'low': close - np.abs(np.random.randn(100))
        }, index=dates)

        plot_data = strategy.get_supertrend_for_plot(df)

        assert 'supertrend' in plot_data
        assert 'trend' in plot_data
        assert 'upper_band' in plot_data
        assert 'lower_band' in plot_data
        assert 'atr' in plot_data

        # All should be Series with same length
        for key, series in plot_data.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(df)

    def test_get_description(self):
        """Test strategy description"""
        strategy = SuperTrendStrategy(period=12, multiplier=2.5)
        description = strategy.get_description()

        assert '12' in description
        assert '2.5' in description
        assert 'SuperTrend' in description

    def test_with_close_only_data(self):
        """Test with DataFrame containing only close prices"""
        strategy = SuperTrendStrategy(period=10)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        # Should handle missing high/low by using close as fallback
        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame is handled gracefully"""
        strategy = SuperTrendStrategy()
        df = pd.DataFrame()

        # Should not crash, but may raise or return empty
        try:
            entries, exits = strategy.generate_signals(df)
            assert len(entries) == 0
            assert len(exits) == 0
        except (ValueError, KeyError, IndexError):
            # Expected - empty data should raise
            pass

    def test_insufficient_data(self):
        """Test with insufficient data for period"""
        strategy = SuperTrendStrategy(period=20)

        # Only 10 data points, less than period
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        close = np.random.randn(10).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        # Should not crash, but signals may be NaN/False
        assert len(entries) == len(df)
        assert len(exits) == len(df)

    def test_atr_with_different_periods(self):
        """Test ATR calculation with various periods"""
        strategy = SuperTrendStrategy(period=10)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        close = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(100)),
            'low': close - np.abs(np.random.randn(100))
        }, index=dates)

        for period in [7, 10, 14, 20]:
            atr = strategy.calculate_atr(df, period=period)
            assert isinstance(atr, pd.Series)
            assert len(atr) == len(df)
