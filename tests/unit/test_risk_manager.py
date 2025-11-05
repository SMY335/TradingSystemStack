"""
Unit tests for Risk Manager

Tests all risk management functionality including:
- VaR calculations (historical, parametric, Monte Carlo)
- CVaR calculations
- Stress testing
- Monte Carlo simulations
- Correlation analysis
- Tail risk metrics
- Risk alerts
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.portfolio.risk_manager import RiskManager, RiskMetrics, StressTestResult


class TestRiskManager:
    """Test suite for RiskManager"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns data"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        returns_data = {
            'Asset_A': np.random.normal(0.0005, 0.02, len(dates)),
            'Asset_B': np.random.normal(0.0008, 0.025, len(dates)),
            'Asset_C': np.random.normal(0.001, 0.03, len(dates)),
        }
        
        return pd.DataFrame(returns_data, index=dates)
    
    @pytest.fixture
    def risk_manager(self, sample_returns):
        """Create RiskManager instance"""
        weights = {'Asset_A': 0.4, 'Asset_B': 0.3, 'Asset_C': 0.3}
        return RiskManager(sample_returns, weights)
    
    def test_initialization(self, sample_returns):
        """Test RiskManager initialization"""
        # Test with custom weights
        weights = {'Asset_A': 0.5, 'Asset_B': 0.3, 'Asset_C': 0.2}
        rm = RiskManager(sample_returns, weights)
        
        assert rm.n_assets == 3
        assert rm.assets == ['Asset_A', 'Asset_B', 'Asset_C']
        assert abs(sum(rm.weights.values()) - 1.0) < 1e-10
        
        # Test with equal weights (None)
        rm_equal = RiskManager(sample_returns)
        assert abs(rm_equal.weights['Asset_A'] - 1/3) < 1e-10
    
    def test_portfolio_returns_calculation(self, risk_manager):
        """Test portfolio returns calculation"""
        port_returns = risk_manager.portfolio_returns
        
        assert isinstance(port_returns, pd.Series)
        assert len(port_returns) == len(risk_manager.returns)
        assert not port_returns.isnull().all()
    
    def test_var_historical(self, risk_manager):
        """Test historical VaR calculation"""
        var_95 = risk_manager.calculate_var_historical(0.95)
        var_99 = risk_manager.calculate_var_historical(0.99)
        
        # VaR should be positive
        assert var_95 > 0
        assert var_99 > 0
        
        # VaR 99% should be higher than VaR 95%
        assert var_99 > var_95
        
        # VaR should be reasonable (between 0% and 50%)
        assert 0 < var_95 < 0.5
        assert 0 < var_99 < 0.5
    
    def test_var_parametric(self, risk_manager):
        """Test parametric VaR calculation"""
        var_95 = risk_manager.calculate_var_parametric(0.95)
        var_99 = risk_manager.calculate_var_parametric(0.99)
        
        assert var_95 > 0
        assert var_99 > 0
        assert var_99 > var_95
    
    def test_var_monte_carlo(self, risk_manager):
        """Test Monte Carlo VaR calculation"""
        var, simulations = risk_manager.calculate_var_monte_carlo(
            confidence_level=0.95,
            n_simulations=1000,
            time_horizon=1
        )
        
        assert var > 0
        assert len(simulations) == 1000
        assert isinstance(simulations, np.ndarray)
    
    def test_cvar_historical(self, risk_manager):
        """Test historical CVaR calculation"""
        cvar_95 = risk_manager.calculate_cvar(0.95, method='historical')
        cvar_99 = risk_manager.calculate_cvar(0.99, method='historical')
        
        # CVaR should be positive
        assert cvar_95 > 0
        assert cvar_99 > 0
        
        # CVaR 99% should be higher than CVaR 95%
        assert cvar_99 > cvar_95
        
        # CVaR should be greater than or equal to VaR
        var_95 = risk_manager.calculate_var_historical(0.95)
        assert cvar_95 >= var_95
    
    def test_cvar_parametric(self, risk_manager):
        """Test parametric CVaR calculation"""
        cvar_95 = risk_manager.calculate_cvar(0.95, method='parametric')
        cvar_99 = risk_manager.calculate_cvar(0.99, method='parametric')
        
        assert cvar_95 > 0
        assert cvar_99 > 0
        assert cvar_99 > cvar_95
    
    def test_stress_test_default_scenarios(self, risk_manager):
        """Test stress testing with default scenarios"""
        results = risk_manager.stress_test()
        
        assert len(results) > 0
        assert all(isinstance(r, StressTestResult) for r in results)
        
        # Check structure
        for result in results:
            assert hasattr(result, 'scenario_name')
            assert hasattr(result, 'portfolio_loss')
            assert hasattr(result, 'loss_percentage')
            assert hasattr(result, 'var_breach')
            assert hasattr(result, 'asset_impacts')
            
            # Loss should be negative for crash scenarios
            if 'Crash' in result.scenario_name:
                assert result.portfolio_loss < 0
    
    def test_stress_test_custom_scenarios(self, risk_manager):
        """Test stress testing with custom scenarios"""
        custom_scenarios = {
            'Custom Crash': {
                'Asset_A': -0.30,
                'Asset_B': -0.25,
                'Asset_C': -0.20
            }
        }
        
        results = risk_manager.stress_test(custom_scenarios)
        
        assert len(results) == 1
        assert results[0].scenario_name == 'Custom Crash'
        assert results[0].portfolio_loss < 0
    
    def test_monte_carlo_simulation(self, risk_manager):
        """Test Monte Carlo simulation"""
        final_values, paths = risk_manager.monte_carlo_simulation(
            n_simulations=100,
            time_horizon=252,
            initial_value=100000
        )
        
        assert len(final_values) == 100
        assert isinstance(paths, pd.DataFrame)
        assert len(paths) == 253  # time_horizon + 1 (initial value)
        
        # Final values should be mostly positive
        assert (final_values > 0).sum() > 90
    
    def test_tail_risk_metrics(self, risk_manager):
        """Test tail risk metrics calculation"""
        metrics = risk_manager.calculate_tail_risk_metrics()
        
        required_keys = [
            'skewness', 'kurtosis', 'excess_kurtosis',
            'jarque_bera_stat', 'jarque_bera_pvalue',
            'worst_1pct', 'worst_5pct', 'best_99pct', 'best_95pct'
        ]
        
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
    
    def test_correlation_analysis(self, risk_manager):
        """Test correlation analysis"""
        # Test correlation matrix
        corr_matrix = risk_manager.correlation_matrix
        
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        
        # Test rolling correlation
        rolling_corr = risk_manager.analyze_correlation_breakdown(window=60)
        assert isinstance(rolling_corr, pd.DataFrame)
    
    def test_max_drawdown(self, risk_manager):
        """Test maximum drawdown calculation"""
        max_dd, drawdown_series = risk_manager.calculate_max_drawdown()
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
        
        assert isinstance(drawdown_series, pd.Series)
        assert len(drawdown_series) == len(risk_manager.portfolio_returns)
        assert (drawdown_series <= 0).all()  # All drawdowns should be <= 0
    
    def test_risk_metrics_comprehensive(self, risk_manager):
        """Test comprehensive risk metrics calculation"""
        metrics = risk_manager.calculate_risk_metrics()
        
        assert isinstance(metrics, RiskMetrics)
        
        # Check all attributes exist
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'var_99')
        assert hasattr(metrics, 'cvar_95')
        assert hasattr(metrics, 'cvar_99')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'sortino_ratio')
        assert hasattr(metrics, 'skewness')
        assert hasattr(metrics, 'kurtosis')
        assert hasattr(metrics, 'worst_loss')
        assert hasattr(metrics, 'best_gain')
        
        # Check reasonable values
        assert metrics.var_95 > 0
        assert metrics.var_99 > metrics.var_95
        assert metrics.volatility > 0
        assert metrics.max_drawdown <= 0
        assert metrics.worst_loss < 0
        assert metrics.best_gain > 0
    
    def test_risk_alerts(self, risk_manager):
        """Test risk alerts generation"""
        alerts = risk_manager.check_risk_alerts(
            var_threshold=0.05,
            drawdown_threshold=-0.20,
            correlation_threshold=0.95
        )
        
        assert 'critical' in alerts
        assert 'warning' in alerts
        assert 'info' in alerts
        
        assert isinstance(alerts['critical'], list)
        assert isinstance(alerts['warning'], list)
        assert isinstance(alerts['info'], list)
    
    def test_risk_report_generation(self, risk_manager):
        """Test risk report generation"""
        report = risk_manager.generate_risk_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'PORTFOLIO RISK ANALYSIS REPORT' in report
        assert 'VaR' in report
        assert 'CVaR' in report
        assert 'Stress Test' in report
    
    def test_different_confidence_levels(self, risk_manager):
        """Test VaR/CVaR at different confidence levels"""
        confidence_levels = [0.90, 0.95, 0.99, 0.999]
        
        prev_var = 0
        for conf in confidence_levels:
            var = risk_manager.calculate_var_historical(conf)
            cvar = risk_manager.calculate_cvar(conf, method='historical')
            
            # VaR should increase with confidence level
            assert var >= prev_var
            prev_var = var
            
            # CVaR should be >= VaR
            assert cvar >= var
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with single asset
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        single_asset_returns = pd.DataFrame({
            'Asset': np.random.normal(0, 0.02, len(dates))
        }, index=dates)
        
        rm = RiskManager(single_asset_returns)
        metrics = rm.calculate_risk_metrics()
        
        assert metrics.var_95 > 0
        assert rm.weights['Asset'] == 1.0
    
    def test_weight_normalization(self, sample_returns):
        """Test that weights are properly normalized"""
        # Test with unnormalized weights
        weights = {'Asset_A': 40, 'Asset_B': 30, 'Asset_C': 30}
        rm = RiskManager(sample_returns, weights)
        
        # Weights should sum to 1
        assert abs(sum(rm.weights.values()) - 1.0) < 1e-10
        
        # Proportions should be maintained
        assert abs(rm.weights['Asset_A'] - 0.4) < 1e-10
        assert abs(rm.weights['Asset_B'] - 0.3) < 1e-10
        assert abs(rm.weights['Asset_C'] - 0.3) < 1e-10


class TestRiskMetrics:
    """Test RiskMetrics dataclass"""
    
    def test_risk_metrics_creation(self):
        """Test RiskMetrics object creation"""
        metrics = RiskMetrics(
            var_95=0.02,
            var_99=0.03,
            cvar_95=0.025,
            cvar_99=0.035,
            volatility=0.15,
            max_drawdown=-0.10,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            skewness=-0.5,
            kurtosis=3.5,
            worst_loss=-0.05,
            best_gain=0.08
        )
        
        assert metrics.var_95 == 0.02
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == -0.10


class TestStressTestResult:
    """Test StressTestResult dataclass"""
    
    def test_stress_test_result_creation(self):
        """Test StressTestResult object creation"""
        result = StressTestResult(
            scenario_name="Test Scenario",
            portfolio_loss=-0.20,
            loss_percentage=-20.0,
            var_breach=True,
            asset_impacts={'Asset_A': -0.10, 'Asset_B': -0.10}
        )
        
        assert result.scenario_name == "Test Scenario"
        assert result.portfolio_loss == -0.20
        assert result.var_breach is True
        assert len(result.asset_impacts) == 2


def test_integration_workflow():
    """Test complete workflow integration"""
    # Generate data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    returns = pd.DataFrame({
        'BTC': np.random.normal(0.001, 0.04, len(dates)),
        'ETH': np.random.normal(0.0012, 0.045, len(dates)),
        'SOL': np.random.normal(0.0015, 0.055, len(dates)),
    }, index=dates)
    
    # Initialize risk manager
    weights = {'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2}
    rm = RiskManager(returns, weights)
    
    # Calculate all metrics
    metrics = rm.calculate_risk_metrics()
    assert metrics is not None
    
    # Run stress tests
    stress_results = rm.stress_test()
    assert len(stress_results) > 0
    
    # Run Monte Carlo
    final_values, paths = rm.monte_carlo_simulation(n_simulations=1000)
    assert len(final_values) == 1000
    
    # Check alerts
    alerts = rm.check_risk_alerts()
    assert alerts is not None
    
    # Generate report
    report = rm.generate_risk_report()
    assert 'PORTFOLIO RISK ANALYSIS REPORT' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
