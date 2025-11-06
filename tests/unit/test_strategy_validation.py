"""
Unit tests for strategy validation
Tests parameter validation for EMAStrategy and other strategies
"""

import pytest
import pandas as pd
import numpy as np
from src.strategies.ema_strategy import EMAStrategy


class TestEMAStrategyValidation:
    """Test EMAStrategy parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        strategy = EMAStrategy(fast_period=12, slow_period=26)
        assert strategy.params['fast_period'] == 12
        assert strategy.params['slow_period'] == 26

    def test_fast_period_less_than_2_rejected(self):
        """Test that fast_period < 2 is rejected"""
        with pytest.raises(ValueError, match="fast_period must be >= 2"):
            EMAStrategy(fast_period=1, slow_period=26)

    def test_slow_period_less_than_2_rejected(self):
        """Test that slow_period < 2 is rejected"""
        with pytest.raises(ValueError, match="slow_period must be >= 2"):
            EMAStrategy(fast_period=12, slow_period=1)

    def test_fast_greater_than_slow_rejected(self):
        """Test that fast_period >= slow_period is rejected"""
        with pytest.raises(ValueError, match="fast_period.*must be < slow_period"):
            EMAStrategy(fast_period=50, slow_period=20)

    def test_fast_equal_to_slow_rejected(self):
        """Test that fast_period == slow_period is rejected"""
        with pytest.raises(ValueError, match="fast_period.*must be < slow_period"):
            EMAStrategy(fast_period=20, slow_period=20)

    def test_non_integer_fast_period_rejected(self):
        """Test that non-integer fast_period is rejected"""
        with pytest.raises(TypeError, match="fast_period must be int"):
            EMAStrategy(fast_period=12.5, slow_period=26)

    def test_non_integer_slow_period_rejected(self):
        """Test that non-integer slow_period is rejected"""
        with pytest.raises(TypeError, match="slow_period must be int"):
            EMAStrategy(fast_period=12, slow_period=26.5)

    def test_string_periods_rejected(self):
        """Test that string periods are rejected"""
        with pytest.raises(TypeError, match="fast_period must be int"):
            EMAStrategy(fast_period="12", slow_period=26)

        with pytest.raises(TypeError, match="slow_period must be int"):
            EMAStrategy(fast_period=12, slow_period="26")

    def test_negative_periods_rejected(self):
        """Test that negative periods are rejected"""
        with pytest.raises(ValueError, match="fast_period must be >= 2"):
            EMAStrategy(fast_period=-5, slow_period=26)

        with pytest.raises(ValueError, match="slow_period must be >= 2"):
            EMAStrategy(fast_period=12, slow_period=-10)

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values
        strategy = EMAStrategy(fast_period=2, slow_period=3)
        assert strategy.params['fast_period'] == 2
        assert strategy.params['slow_period'] == 3

        # Large values
        strategy = EMAStrategy(fast_period=50, slow_period=200)
        assert strategy.params['fast_period'] == 50
        assert strategy.params['slow_period'] == 200

    def test_default_values(self):
        """Test that default values are valid"""
        strategy = EMAStrategy()
        assert strategy.params['fast_period'] == 12
        assert strategy.params['slow_period'] == 26
        assert strategy.params['fast_period'] < strategy.params['slow_period']

    def test_generate_signals_with_valid_data(self):
        """Test that signal generation works with valid data"""
        strategy = EMAStrategy(fast_period=5, slow_period=10)

        # Create sample data with enough points
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(50).cumsum() + 100
        }, index=dates)

        entries, exits = strategy.generate_signals(df)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert len(entries) == len(df)
        assert len(exits) == len(df)

    def test_get_description(self):
        """Test that get_description returns correct format"""
        strategy = EMAStrategy(fast_period=12, slow_period=26)
        description = strategy.get_description()

        assert "EMA(12)" in description
        assert "EMA(26)" in description

    def test_get_param_schema(self):
        """Test that parameter schema is correct"""
        strategy = EMAStrategy()
        schema = strategy.get_param_schema()

        assert 'fast_period' in schema
        assert 'slow_period' in schema

        # Check fast_period schema
        assert schema['fast_period']['type'] == 'int'
        assert schema['fast_period']['min'] == 5
        assert schema['fast_period']['max'] == 50

        # Check slow_period schema
        assert schema['slow_period']['type'] == 'int'
        assert schema['slow_period']['min'] == 20
        assert schema['slow_period']['max'] == 200


class TestStrategyRobustness:
    """Test strategy robustness with edge cases"""

    def test_handles_empty_dataframe(self):
        """Test that empty DataFrame is handled"""
        strategy = EMAStrategy()
        df = pd.DataFrame()

        # Should either raise or return empty signals
        try:
            entries, exits = strategy.generate_signals(df)
            assert len(entries) == 0
            assert len(exits) == 0
        except (ValueError, KeyError):
            # Also acceptable - empty data should raise
            pass

    def test_handles_insufficient_data(self):
        """Test behavior with insufficient data points"""
        strategy = EMAStrategy(fast_period=50, slow_period=100)

        # Only 10 data points - not enough for slow EMA
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })

        entries, exits = strategy.generate_signals(df)

        # Signals should be valid even if mostly NaN/False
        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
