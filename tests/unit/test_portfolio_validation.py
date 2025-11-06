"""
Unit tests for portfolio module validation
Tests validation for RiskManager, PortfolioManager, and PortfolioOptimizer
"""

import pytest
import pandas as pd
import numpy as np
from src.portfolio.risk_manager import RiskManager
from src.portfolio.optimizer import PortfolioOptimizer


class TestRiskManagerValidation:
    """Test RiskManager parameter validation"""

    def create_sample_returns(self, n_assets=3, n_periods=100):
        """Helper to create sample returns data"""
        dates = pd.date_range('2024-01-01', periods=n_periods, freq='D')
        data = {}
        for i in range(n_assets):
            data[f'ASSET{i+1}'] = np.random.randn(n_periods) * 0.02
        return pd.DataFrame(data, index=dates)

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': 0.4, 'ASSET2': 0.3, 'ASSET3': 0.3}

        manager = RiskManager(
            returns=returns,
            weights=weights,
            risk_free_rate=0.02
        )

        assert manager.risk_free_rate == 0.02
        assert len(manager.weights) == 3

    def test_empty_returns_rejected(self):
        """Test that empty returns DataFrame is rejected"""
        returns = pd.DataFrame()
        weights = {'ASSET1': 1.0}

        with pytest.raises(ValueError, match="returns DataFrame cannot be empty"):
            RiskManager(returns=returns, weights=weights)

    def test_insufficient_data_points_rejected(self):
        """Test that < 2 data points is rejected"""
        returns = self.create_sample_returns(n_periods=1)
        weights = {'ASSET1': 0.5, 'ASSET2': 0.5}

        with pytest.raises(ValueError, match="Need at least 2 data points"):
            RiskManager(returns=returns, weights=weights)

    def test_non_dataframe_returns_rejected(self):
        """Test that non-DataFrame returns is rejected"""
        weights = {'ASSET1': 1.0}

        with pytest.raises(TypeError, match="returns must be pandas DataFrame"):
            RiskManager(returns="not a dataframe", weights=weights)

    def test_empty_weights_rejected(self):
        """Test that empty weights dict is rejected"""
        returns = self.create_sample_returns()

        with pytest.raises(ValueError, match="weights cannot be empty"):
            RiskManager(returns=returns, weights={})

    def test_non_dict_weights_rejected(self):
        """Test that non-dict weights is rejected"""
        returns = self.create_sample_returns()

        with pytest.raises(TypeError, match="weights must be dict"):
            RiskManager(returns=returns, weights=[0.5, 0.5])

    def test_negative_weight_rejected(self):
        """Test that negative weights are rejected"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': -0.5, 'ASSET2': 1.5}

        with pytest.raises(ValueError, match="cannot be negative"):
            RiskManager(returns=returns, weights=weights)

    def test_weight_above_one_rejected(self):
        """Test that weights > 1 are rejected"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': 1.5, 'ASSET2': -0.5}

        with pytest.raises(ValueError, match="cannot exceed 1"):
            RiskManager(returns=returns, weights=weights)

    def test_non_numeric_weight_rejected(self):
        """Test that non-numeric weights are rejected"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': "0.5", 'ASSET2': 0.5}

        with pytest.raises(TypeError, match="must be numeric"):
            RiskManager(returns=returns, weights=weights)

    def test_weights_not_summing_to_one_normalized(self):
        """Test that weights not summing to 1.0 are auto-normalized"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': 0.4, 'ASSET2': 0.3, 'ASSET3': 0.2}  # Sums to 0.9

        # Should not raise, should normalize
        manager = RiskManager(returns=returns, weights=weights)
        assert manager is not None

    def test_invalid_risk_free_rate_rejected(self):
        """Test that invalid risk_free_rate is rejected"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': 0.5, 'ASSET2': 0.5}

        # Below -1
        with pytest.raises(ValueError, match="risk_free_rate must be between -1 and 1"):
            RiskManager(returns=returns, weights=weights, risk_free_rate=-1.5)

        # Above 1
        with pytest.raises(ValueError, match="risk_free_rate must be between -1 and 1"):
            RiskManager(returns=returns, weights=weights, risk_free_rate=1.5)

    def test_invalid_confidence_level_rejected(self):
        """Test that invalid confidence levels are rejected"""
        returns = self.create_sample_returns()
        weights = {'ASSET1': 0.5, 'ASSET2': 0.5}

        # Confidence level = 0
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            RiskManager(returns=returns, weights=weights, confidence_levels=[0.0])

        # Confidence level = 1
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            RiskManager(returns=returns, weights=weights, confidence_levels=[1.0])

        # Confidence level > 1
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            RiskManager(returns=returns, weights=weights, confidence_levels=[1.5])

    def test_calculate_risk_metrics(self):
        """Test that risk metrics calculation works"""
        returns = self.create_sample_returns(n_periods=252)
        weights = {'ASSET1': 0.4, 'ASSET2': 0.3, 'ASSET3': 0.3}

        manager = RiskManager(returns=returns, weights=weights)
        metrics = manager.calculate_risk_metrics()

        # Check all expected fields exist
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'var_99')
        assert hasattr(metrics, 'cvar_95')
        assert hasattr(metrics, 'cvar_99')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')

    def test_max_drawdown_no_division_by_zero(self):
        """Test that max drawdown calculation doesn't divide by zero"""
        # Create returns that could cause issues
        returns = pd.DataFrame({
            'ASSET1': [0.0] * 100  # All zeros
        })
        weights = {'ASSET1': 1.0}

        manager = RiskManager(returns=returns, weights=weights)

        # Should not raise division by zero error
        max_dd = manager.calculate_max_drawdown()
        assert isinstance(max_dd, (int, float))
        assert not np.isnan(max_dd)
        assert not np.isinf(max_dd)


class TestPortfolioOptimizerValidation:
    """Test PortfolioOptimizer parameter validation"""

    def test_valid_initialization(self):
        """Test that valid parameters are accepted"""
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            frequency=252,
            alpha=0.05
        )

        assert optimizer.risk_free_rate == 0.02
        assert optimizer.frequency == 252
        assert optimizer.alpha == 0.05

    def test_invalid_risk_free_rate_rejected(self):
        """Test that invalid risk_free_rate is rejected"""
        with pytest.raises(ValueError, match="risk_free_rate must be between -1 and 1"):
            PortfolioOptimizer(risk_free_rate=-1.5)

        with pytest.raises(ValueError, match="risk_free_rate must be between -1 and 1"):
            PortfolioOptimizer(risk_free_rate=1.5)

    def test_non_numeric_risk_free_rate_rejected(self):
        """Test that non-numeric risk_free_rate is rejected"""
        with pytest.raises(TypeError, match="risk_free_rate must be numeric"):
            PortfolioOptimizer(risk_free_rate="0.02")

    def test_negative_frequency_rejected(self):
        """Test that negative frequency is rejected"""
        with pytest.raises(ValueError, match="frequency must be positive"):
            PortfolioOptimizer(frequency=-252)

    def test_zero_frequency_rejected(self):
        """Test that zero frequency is rejected"""
        with pytest.raises(ValueError, match="frequency must be positive"):
            PortfolioOptimizer(frequency=0)

    def test_frequency_above_365_rejected(self):
        """Test that frequency > 365 is rejected"""
        with pytest.raises(ValueError, match="frequency cannot exceed 365"):
            PortfolioOptimizer(frequency=500)

    def test_non_integer_frequency_rejected(self):
        """Test that non-integer frequency is rejected"""
        with pytest.raises(TypeError, match="frequency must be int"):
            PortfolioOptimizer(frequency=252.5)

    def test_alpha_zero_rejected(self):
        """Test that alpha = 0 is rejected"""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            PortfolioOptimizer(alpha=0.0)

    def test_alpha_one_rejected(self):
        """Test that alpha = 1 is rejected"""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            PortfolioOptimizer(alpha=1.0)

    def test_alpha_above_one_rejected(self):
        """Test that alpha > 1 is rejected"""
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            PortfolioOptimizer(alpha=1.5)

    def test_non_numeric_alpha_rejected(self):
        """Test that non-numeric alpha is rejected"""
        with pytest.raises(TypeError, match="alpha must be numeric"):
            PortfolioOptimizer(alpha="0.05")

    def test_boundary_values(self):
        """Test boundary values are accepted"""
        # Minimum valid values
        optimizer = PortfolioOptimizer(
            risk_free_rate=-0.99,
            frequency=1,
            alpha=0.001
        )
        assert optimizer.risk_free_rate == -0.99
        assert optimizer.frequency == 1
        assert optimizer.alpha == 0.001

        # Maximum valid values
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.99,
            frequency=365,
            alpha=0.999
        )
        assert optimizer.risk_free_rate == 0.99
        assert optimizer.frequency == 365
        assert optimizer.alpha == 0.999

    def test_default_values(self):
        """Test that default values are valid"""
        optimizer = PortfolioOptimizer()

        assert optimizer.risk_free_rate == 0.0
        assert optimizer.frequency == 252
        assert optimizer.alpha == 0.05


class TestFinancialCalculationRobustness:
    """Test robustness of financial calculations"""

    def test_var_with_zero_volatility(self):
        """Test VaR calculation with zero volatility data"""
        # All returns are the same (zero volatility)
        returns = pd.DataFrame({
            'ASSET1': [0.01] * 100
        })
        weights = {'ASSET1': 1.0}

        manager = RiskManager(returns=returns, weights=weights)

        # Should not crash
        var_95 = manager.calculate_var(confidence=0.95)
        assert isinstance(var_95, (int, float))
        assert not np.isnan(var_95)

    def test_stress_test_completeness(self):
        """Test that stress tests complete without errors"""
        returns = pd.DataFrame({
            'ASSET1': np.random.randn(100) * 0.02,
            'ASSET2': np.random.randn(100) * 0.03
        })
        weights = {'ASSET1': 0.6, 'ASSET2': 0.4}

        manager = RiskManager(returns=returns, weights=weights)

        # Should not crash
        stress_results = manager.stress_test()

        assert isinstance(stress_results, list)
        assert len(stress_results) > 0

        for result in stress_results:
            assert hasattr(result, 'scenario_name')
            assert hasattr(result, 'portfolio_return')
            assert hasattr(result, 'portfolio_loss')
