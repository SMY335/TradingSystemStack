"""
Unit tests for IchimokuStrategy validation
Tests all parameter validation for the Ichimoku Cloud strategy
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.ichimoku_strategy import IchimokuStrategy


class TestIchimokuValidation:
    """Test IchimokuStrategy parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        assert strategy.params['tenkan_period'] == 9
        assert strategy.params['kijun_period'] == 26
        assert strategy.params['senkou_b_period'] == 52

    def test_default_parameters(self):
        """Test default parameter values"""
        strategy = IchimokuStrategy()
        assert strategy.params['tenkan_period'] == 9
        assert strategy.params['kijun_period'] == 26
        assert strategy.params['senkou_b_period'] == 52

    def test_tenkan_not_int_rejected(self):
        """Test that non-integer tenkan_period is rejected"""
        with pytest.raises(TypeError, match="tenkan_period must be int"):
            IchimokuStrategy(tenkan_period=9.5)

    def test_tenkan_too_small_rejected(self):
        """Test that tenkan_period < 1 is rejected"""
        with pytest.raises(ValueError, match="tenkan_period must be >= 1"):
            IchimokuStrategy(tenkan_period=0)

    def test_tenkan_too_large_rejected(self):
        """Test that tenkan_period > 50 is rejected"""
        with pytest.raises(ValueError, match="tenkan_period cannot exceed 50"):
            IchimokuStrategy(tenkan_period=51)

    def test_kijun_not_int_rejected(self):
        """Test that non-integer kijun_period is rejected"""
        with pytest.raises(TypeError, match="kijun_period must be int"):
            IchimokuStrategy(kijun_period=26.5)

    def test_kijun_too_small_rejected(self):
        """Test that kijun_period < 1 is rejected"""
        with pytest.raises(ValueError, match="kijun_period must be >= 1"):
            IchimokuStrategy(kijun_period=0)

    def test_kijun_too_large_rejected(self):
        """Test that kijun_period > 100 is rejected"""
        with pytest.raises(ValueError, match="kijun_period cannot exceed 100"):
            IchimokuStrategy(kijun_period=101)

    def test_senkou_b_not_int_rejected(self):
        """Test that non-integer senkou_b_period is rejected"""
        with pytest.raises(TypeError, match="senkou_b_period must be int"):
            IchimokuStrategy(senkou_b_period=52.5)

    def test_senkou_b_too_small_rejected(self):
        """Test that senkou_b_period < 1 is rejected"""
        with pytest.raises(ValueError, match="senkou_b_period must be >= 1"):
            IchimokuStrategy(senkou_b_period=0)

    def test_senkou_b_too_large_rejected(self):
        """Test that senkou_b_period > 200 is rejected"""
        with pytest.raises(ValueError, match="senkou_b_period cannot exceed 200"):
            IchimokuStrategy(senkou_b_period=201)

    def test_tenkan_greater_than_kijun_rejected(self):
        """Test that tenkan_period >= kijun_period is rejected"""
        with pytest.raises(ValueError, match="tenkan_period.*should be < kijun_period"):
            IchimokuStrategy(tenkan_period=30, kijun_period=20)

        with pytest.raises(ValueError, match="tenkan_period.*should be < kijun_period"):
            IchimokuStrategy(tenkan_period=26, kijun_period=26)

    def test_kijun_greater_than_senkou_b_rejected(self):
        """Test that kijun_period >= senkou_b_period is rejected"""
        with pytest.raises(ValueError, match="kijun_period.*should be < senkou_b_period"):
            IchimokuStrategy(tenkan_period=9, kijun_period=60, senkou_b_period=50)

        with pytest.raises(ValueError, match="kijun_period.*should be < senkou_b_period"):
            IchimokuStrategy(tenkan_period=9, kijun_period=52, senkou_b_period=52)

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values (respecting ordering)
        strategy = IchimokuStrategy(tenkan_period=1, kijun_period=2, senkou_b_period=3)
        assert strategy.params['tenkan_period'] == 1
        assert strategy.params['kijun_period'] == 2
        assert strategy.params['senkou_b_period'] == 3

        # Maximum valid values (respecting ordering)
        strategy = IchimokuStrategy(tenkan_period=50, kijun_period=100, senkou_b_period=200)
        assert strategy.params['tenkan_period'] == 50
        assert strategy.params['kijun_period'] == 100
        assert strategy.params['senkou_b_period'] == 200

    def test_param_schema(self):
        """Test that parameter schema is correctly defined"""
        strategy = IchimokuStrategy()
        schema = strategy.get_param_schema()

        assert 'tenkan_period' in schema
        assert schema['tenkan_period']['type'] == 'int'
        assert schema['tenkan_period']['min'] == 7
        assert schema['tenkan_period']['max'] == 12
        assert schema['tenkan_period']['default'] == 9

        assert 'kijun_period' in schema
        assert schema['kijun_period']['type'] == 'int'
        assert schema['kijun_period']['min'] == 20
        assert schema['kijun_period']['max'] == 30
        assert schema['kijun_period']['default'] == 26

        assert 'senkou_b_period' in schema
        assert schema['senkou_b_period']['type'] == 'int'
        assert schema['senkou_b_period']['min'] == 40
        assert schema['senkou_b_period']['max'] == 60
        assert schema['senkou_b_period']['default'] == 52


class TestIchimokuSignals:
    """Test IchimokuStrategy signal generation and calculations"""

    def test_generate_signals_with_valid_data(self):
        """Test signal generation with valid OHLC data"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        # Create sample OHLC data
        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)
        assert len(exits) == len(df)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_calculate_ichimoku_components(self):
        """Test Ichimoku component calculations"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        ichimoku = strategy.calculate_ichimoku(df)

        # Check all 5 components exist
        assert 'tenkan_sen' in ichimoku
        assert 'kijun_sen' in ichimoku
        assert 'senkou_span_a' in ichimoku
        assert 'senkou_span_b' in ichimoku
        assert 'chikou_span' in ichimoku

        # All should be Series with same length as input
        for key, series in ichimoku.items():
            assert isinstance(series, pd.Series)
            assert len(series) == len(df)

    def test_tenkan_sen_calculation(self):
        """Test Tenkan-sen (Conversion Line) is calculated correctly"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        ichimoku = strategy.calculate_ichimoku(df)
        tenkan = ichimoku['tenkan_sen']

        # Tenkan should be between high and low of recent periods
        valid_data = tenkan.iloc[9:].dropna()
        if len(valid_data) > 0:
            for i in range(9, len(df)):
                if not pd.isna(tenkan.iloc[i]):
                    recent_high = df['high'].iloc[max(0, i-8):i+1].max()
                    recent_low = df['low'].iloc[max(0, i-8):i+1].min()
                    assert tenkan.iloc[i] >= recent_low - 0.01  # Small tolerance for floating point
                    assert tenkan.iloc[i] <= recent_high + 0.01

    def test_get_cloud_color(self):
        """Test cloud color determination"""
        strategy = IchimokuStrategy()

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        cloud_color = strategy.get_cloud_color(df)

        assert isinstance(cloud_color, pd.Series)
        assert len(cloud_color) == len(df)

        # Cloud color should be -1 (bearish), 0 (neutral), or 1 (bullish)
        valid_colors = cloud_color.dropna()
        if len(valid_colors) > 0:
            assert set(valid_colors.unique()).issubset({-1, 0, 1})

    def test_price_position_relative_to_cloud(self):
        """Test price position relative to cloud"""
        strategy = IchimokuStrategy()

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        position = strategy.price_position_relative_to_cloud(df)

        assert isinstance(position, pd.Series)
        assert len(position) == len(df)

        # Position should be -1 (below), 0 (in cloud), or 1 (above)
        valid_positions = position.dropna()
        if len(valid_positions) > 0:
            assert set(valid_positions.unique()).issubset({-1, 0, 1})

    def test_get_description(self):
        """Test strategy description"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)
        description = strategy.get_description()

        assert '9' in description
        assert '26' in description
        assert '52' in description
        assert 'Ichimoku' in description

    def test_with_close_only_data(self):
        """Test with DataFrame containing only close prices"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(150).cumsum() + 100
        }, index=dates)

        # Should handle missing high/low by using close as fallback
        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame is handled gracefully"""
        strategy = IchimokuStrategy()
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
        """Test with insufficient data for longest period"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        # Only 30 data points, less than senkou_b_period
        dates = pd.date_range('2024-01-01', periods=30, freq='1H')
        close = np.random.randn(30).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        # Should not crash, but signals may be NaN/False
        assert len(entries) == len(df)
        assert len(exits) == len(df)

    def test_chikou_span_shift(self):
        """Test that Chikou Span is correctly shifted backward"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1
        }, index=dates)

        ichimoku = strategy.calculate_ichimoku(df)
        chikou = ichimoku['chikou_span']

        # Chikou should be close shifted back by kijun_period
        # The last 26 values should be NaN (shifted backward)
        assert chikou.iloc[-26:].isna().all()

    def test_senkou_span_shift(self):
        """Test that Senkou Spans are correctly shifted forward"""
        strategy = IchimokuStrategy(tenkan_period=9, kijun_period=26, senkou_b_period=52)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + 1,
            'low': close - 1
        }, index=dates)

        ichimoku = strategy.calculate_ichimoku(df)
        senkou_a = ichimoku['senkou_span_a']
        senkou_b = ichimoku['senkou_span_b']

        # Senkou spans should be shifted forward by kijun_period (26)
        # The first 26 values should be NaN
        assert senkou_a.iloc[:26].isna().all()
        assert senkou_b.iloc[:26].isna().all()

    def test_custom_periods(self):
        """Test with non-standard custom periods"""
        strategy = IchimokuStrategy(tenkan_period=7, kijun_period=22, senkou_b_period=44)

        dates = pd.date_range('2024-01-01', periods=150, freq='1H')
        close = np.random.randn(150).cumsum() + 100
        df = pd.DataFrame({
            'close': close,
            'high': close + np.abs(np.random.randn(150)),
            'low': close - np.abs(np.random.randn(150))
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)
        assert len(exits) == len(df)
