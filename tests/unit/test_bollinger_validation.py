"""
Unit tests for BollingerBandsStrategy validation
Tests all parameter validation for the Bollinger Bands strategy
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.bollinger_strategy import BollingerBandsStrategy


class TestBollingerBandsValidation:
    """Test BollingerBandsStrategy parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
        assert strategy.params['period'] == 20
        assert strategy.params['std_dev'] == 2.0

    def test_default_parameters(self):
        """Test default parameter values"""
        strategy = BollingerBandsStrategy()
        assert strategy.params['period'] == 20
        assert strategy.params['std_dev'] == 2.0

    def test_period_not_int_rejected(self):
        """Test that non-integer period is rejected"""
        with pytest.raises(TypeError, match="period must be int"):
            BollingerBandsStrategy(period=20.5)

    def test_period_too_small_rejected(self):
        """Test that period < 2 is rejected"""
        with pytest.raises(ValueError, match="period must be >= 2"):
            BollingerBandsStrategy(period=1)

    def test_period_too_large_rejected(self):
        """Test that period > 200 is rejected"""
        with pytest.raises(ValueError, match="period cannot exceed 200"):
            BollingerBandsStrategy(period=201)

    def test_std_dev_not_numeric_rejected(self):
        """Test that non-numeric std_dev is rejected"""
        with pytest.raises(TypeError, match="std_dev must be numeric"):
            BollingerBandsStrategy(std_dev="2.0")

    def test_std_dev_non_positive_rejected(self):
        """Test that std_dev <= 0 is rejected"""
        with pytest.raises(ValueError, match="std_dev must be positive"):
            BollingerBandsStrategy(std_dev=0)

        with pytest.raises(ValueError, match="std_dev must be positive"):
            BollingerBandsStrategy(std_dev=-1.0)

    def test_std_dev_too_large_rejected(self):
        """Test that std_dev > 10 is rejected"""
        with pytest.raises(ValueError, match="std_dev cannot exceed 10"):
            BollingerBandsStrategy(std_dev=10.5)

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values
        strategy = BollingerBandsStrategy(period=2, std_dev=0.1)
        assert strategy.params['period'] == 2
        assert strategy.params['std_dev'] == 0.1

        # Maximum valid values
        strategy = BollingerBandsStrategy(period=200, std_dev=10.0)
        assert strategy.params['period'] == 200
        assert strategy.params['std_dev'] == 10.0

    def test_param_schema(self):
        """Test that parameter schema is correctly defined"""
        strategy = BollingerBandsStrategy()
        schema = strategy.get_param_schema()

        assert 'period' in schema
        assert schema['period']['type'] == 'int'
        assert schema['period']['min'] == 10
        assert schema['period']['max'] == 50
        assert schema['period']['default'] == 20

        assert 'std_dev' in schema
        assert schema['std_dev']['type'] == 'float'
        assert schema['std_dev']['min'] == 1.0
        assert schema['std_dev']['max'] == 4.0


class TestBollingerBandsSignals:
    """Test BollingerBandsStrategy signal generation"""

    def test_generate_signals_with_valid_data(self):
        """Test signal generation with valid price data"""
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)

        # Create sample data with trend
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)
        assert len(exits) == len(df)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_calculate_bands(self):
        """Test Bollinger Bands calculation"""
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        bands = strategy.calculate_bands(df)

        assert 'upper' in bands
        assert 'middle' in bands
        assert 'lower' in bands
        assert 'width' in bands

        # Upper should be above middle, middle above lower
        valid_data = bands['upper'].iloc[20:].dropna()
        if len(valid_data) > 0:
            assert (bands['upper'].iloc[20:] > bands['middle'].iloc[20:]).all()
            assert (bands['middle'].iloc[20:] > bands['lower'].iloc[20:]).all()

    def test_detect_squeeze(self):
        """Test Bollinger Band squeeze detection"""
        strategy = BollingerBandsStrategy(period=20, std_dev=2.0)

        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        squeeze = strategy.detect_squeeze(df)

        assert isinstance(squeeze, pd.Series)
        assert squeeze.dtype == bool
        assert len(squeeze) == len(df)

    def test_get_description(self):
        """Test strategy description"""
        strategy = BollingerBandsStrategy(period=25, std_dev=2.5)
        description = strategy.get_description()

        assert '25' in description
        assert '2.5' in description
        assert 'Bollinger' in description

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrame is handled gracefully"""
        strategy = BollingerBandsStrategy()
        df = pd.DataFrame()

        # Should not crash, but may raise or return empty
        try:
            entries, exits = strategy.generate_signals(df)
            assert len(entries) == 0
            assert len(exits) == 0
        except (ValueError, KeyError):
            # Expected - empty data should raise
            pass

    def test_insufficient_data(self):
        """Test with insufficient data for period"""
        strategy = BollingerBandsStrategy(period=20)

        # Only 10 data points, less than period
        dates = pd.date_range('2024-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(10).cumsum() + 100
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        # Should not crash, but signals may be NaN/False
        assert len(entries) == len(df)
        assert len(exits) == len(df)
