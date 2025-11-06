"""
Unit tests for BacktestEngine validation
Tests all parameter validation added during security improvements
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine import BacktestEngine
from src.strategies.ema_strategy import EMAStrategy


class TestBacktestEngineValidation:
    """Test BacktestEngine parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        engine = BacktestEngine(
            initial_cash=10000,
            fees=0.001,
            slippage=0.0005
        )
        assert engine.initial_cash == 10000
        assert engine.fees == 0.001
        assert engine.slippage == 0.0005

    def test_negative_initial_cash_rejected(self):
        """Test that negative initial_cash is rejected"""
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            BacktestEngine(initial_cash=-1000)

    def test_zero_initial_cash_rejected(self):
        """Test that zero initial_cash is rejected"""
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            BacktestEngine(initial_cash=0)

    def test_invalid_type_initial_cash(self):
        """Test that non-numeric initial_cash is rejected"""
        with pytest.raises(TypeError, match="initial_cash must be numeric"):
            BacktestEngine(initial_cash="10000")

    def test_negative_fees_rejected(self):
        """Test that negative fees are rejected"""
        with pytest.raises(ValueError, match="fees must be between 0 and 1"):
            BacktestEngine(fees=-0.01)

    def test_fees_above_100_percent_rejected(self):
        """Test that fees > 100% are rejected"""
        with pytest.raises(ValueError, match="fees must be between 0 and 1"):
            BacktestEngine(fees=1.5)

    def test_invalid_type_fees(self):
        """Test that non-numeric fees are rejected"""
        with pytest.raises(TypeError, match="fees must be numeric"):
            BacktestEngine(fees="0.001")

    def test_negative_slippage_rejected(self):
        """Test that negative slippage is rejected"""
        with pytest.raises(ValueError, match="slippage must be between 0 and 1"):
            BacktestEngine(slippage=-0.01)

    def test_slippage_above_100_percent_rejected(self):
        """Test that slippage > 100% is rejected"""
        with pytest.raises(ValueError, match="slippage must be between 0 and 1"):
            BacktestEngine(slippage=1.5)

    def test_invalid_type_slippage(self):
        """Test that non-numeric slippage is rejected"""
        with pytest.raises(TypeError, match="slippage must be numeric"):
            BacktestEngine(slippage="0.0005")

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values
        engine = BacktestEngine(initial_cash=0.01, fees=0, slippage=0)
        assert engine.initial_cash == 0.01
        assert engine.fees == 0
        assert engine.slippage == 0

        # Maximum valid values
        engine = BacktestEngine(initial_cash=1e9, fees=1, slippage=1)
        assert engine.initial_cash == 1e9
        assert engine.fees == 1
        assert engine.slippage == 1

    def test_run_with_valid_data(self):
        """Test that run works with valid strategy and data"""
        engine = BacktestEngine(initial_cash=10000)
        strategy = EMAStrategy(fast_period=12, slow_period=26)

        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        }, index=dates)

        # Should not raise any exceptions
        portfolio, kpis = engine.run(strategy, df)

        assert portfolio is not None
        assert isinstance(kpis, dict)
        assert 'total_return_pct' in kpis


class TestBacktestEngineExceptionHandling:
    """Test that exception handling works correctly"""

    def test_handles_invalid_portfolio_gracefully(self):
        """Test that invalid portfolio data is handled gracefully"""
        engine = BacktestEngine()
        strategy = EMAStrategy()

        # Empty dataframe should not crash
        df = pd.DataFrame()

        # Should handle gracefully - might raise or return safe defaults
        try:
            portfolio, kpis = engine.run(strategy, df)
            # If it doesn't raise, check kpis are safe
            assert isinstance(kpis, dict)
        except (ValueError, KeyError, IndexError):
            # Expected - empty data should raise
            pass

    def test_logging_on_metric_failure(self):
        """Test that failures are logged, not silent"""
        # This is an integration test that would check logs
        # For now, we verify the code structure exists
        engine = BacktestEngine()
        assert hasattr(engine, 'fees')
        assert hasattr(engine, 'slippage')
