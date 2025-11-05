"""
Unit tests for Performance Attribution

Tests all performance attribution functionality including:
- Brinson attribution
- Factor attribution
- Asset contributions
- Risk attribution
- Time-weighted and money-weighted returns
- Rolling attribution
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.portfolio.performance_attribution import (
    PerformanceAttributor,
    AttributionResult,
    RiskAttribution
)


class TestPerformanceAttributor:
    """Test suite for PerformanceAttributor"""
    
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
    def attributor(self, sample_returns):
        """Create PerformanceAttributor instance"""
        portfolio_weights = {'Asset_A': 0.4, 'Asset_B': 0.3, 'Asset_C': 0.3}
        benchmark_weights = {'Asset_A': 1/3, 'Asset_B': 1/3, 'Asset_C': 1/3}
        return PerformanceAttributor(sample_returns, portfolio_weights, benchmark_weights)
    
    def test_initialization(self, sample_returns):
        """Test PerformanceAttributor initialization"""
        portfolio_weights = {'Asset_A': 0.5, 'Asset_B': 0.3, 'Asset_C': 0.2}
        
        # With custom benchmark
        benchmark_weights = {'Asset_A': 0.4, 'Asset_B': 0.4, 'Asset_C': 0.2}
        attr = PerformanceAttributor(sample_returns, portfolio_weights, benchmark_weights)
        
        assert attr.n_assets == 3
        assert attr.assets == ['Asset_A', 'Asset_B', 'Asset_C']
        assert abs(sum(attr.portfolio_weights.values()) - 1.0) < 1e-10
        assert abs(sum(attr.benchmark_weights.values()) - 1.0) < 1e-10
        
        # With equal weight benchmark (None)
        attr_equal = PerformanceAttributor(sample_returns, portfolio_weights)
        assert abs(attr_equal.benchmark_weights['Asset_A'] - 1/3) < 1e-10
    
    def test_weighted_returns_calculation(self, attributor):
        """Test weighted returns calculation"""
        port_returns = attributor.portfolio_returns
        bench_returns = attributor.benchmark_returns
        
        assert isinstance(port_returns, pd.Series)
        assert isinstance(bench_returns, pd.Series)
        assert len(port_returns) == len(attributor.returns)
        assert len(bench_returns) == len(attributor.returns)
    
    def test_twr_calculation(self, attributor):
        """Test Time-Weighted Return calculation"""
        twr = attributor.calculate_twr()
        
        assert isinstance(twr, float)
        assert -1 < twr < 10  # Reasonable bounds
        
        # Test with date range
        twr_partial = attributor.calculate_twr(
            start_date='2021-01-01',
            end_date='2022-12-31'
        )
        assert isinstance(twr_partial, float)
    
    def test_brinson_attribution(self, attributor):
        """Test Brinson-Fachler attribution"""
        attribution = attributor.brinson_attribution()
        
        assert isinstance(attribution, AttributionResult)
        
        # Check all attributes exist
        assert hasattr(attribution, 'total_return')
        assert hasattr(attribution, 'benchmark_return')
        assert hasattr(attribution, 'active_return')
        assert hasattr(attribution, 'allocation_effect')
        assert hasattr(attribution, 'selection_effect')
        assert hasattr(attribution, 'interaction_effect')
        assert hasattr(attribution, 'asset_contributions')
        
        # Active return should equal portfolio - benchmark
        expected_active = attribution.total_return - attribution.benchmark_return
        assert abs(attribution.active_return - expected_active) < 1e-6
        
        # Asset contributions should sum to total return
        total_contrib = sum(attribution.asset_contributions.values())
        assert abs(total_contrib - attribution.total_return) < 1e-6
    
    def test_brinson_components_sum(self, attributor):
        """Test that Brinson components sum correctly"""
        attribution = attributor.brinson_attribution()
        
        # Allocation + Selection + Interaction should approximate Active Return
        components_sum = (
            attribution.allocation_effect +
            attribution.selection_effect +
            attribution.interaction_effect
        )
        
        # Allow some tolerance due to calculation method
        assert abs(components_sum - attribution.active_return) < 0.1
    
    def test_factor_attribution(self, attributor):
        """Test factor attribution"""
        factor_contrib = attributor.factor_attribution()
        
        assert isinstance(factor_contrib, dict)
        assert 'Alpha' in factor_contrib
        assert 'Market' in factor_contrib
        
        # All values should be finite
        for value in factor_contrib.values():
            assert np.isfinite(value)
    
    def test_factor_attribution_custom_factors(self, attributor):
        """Test factor attribution with custom factors"""
        # Create custom factors
        dates = attributor.returns.index
        factors = pd.DataFrame({
            'Factor1': np.random.normal(0, 0.01, len(dates)),
            'Factor2': np.random.normal(0, 0.01, len(dates))
        }, index=dates)
        
        factor_contrib = attributor.factor_attribution(factors)
        
        assert 'Alpha' in factor_contrib
        assert 'Factor1' in factor_contrib
        assert 'Factor2' in factor_contrib
    
    def test_asset_contributions_full(self, attributor):
        """Test full period asset contributions"""
        contrib = attributor.calculate_asset_contributions(period='full')
        
        assert isinstance(contrib, pd.DataFrame)
        assert len(contrib) == 1
        assert list(contrib.columns) == attributor.assets
    
    def test_asset_contributions_daily(self, attributor):
        """Test daily asset contributions"""
        contrib = attributor.calculate_asset_contributions(period='daily')
        
        assert isinstance(contrib, pd.DataFrame)
        assert len(contrib) == len(attributor.returns)
        assert list(contrib.columns) == attributor.assets
    
    def test_asset_contributions_monthly(self, attributor):
        """Test monthly asset contributions"""
        contrib = attributor.calculate_asset_contributions(period='monthly')
        
        assert isinstance(contrib, pd.DataFrame)
        assert list(contrib.columns) == attributor.assets
    
    def test_risk_attribution(self, attributor):
        """Test risk attribution analysis"""
        risk_attr = attributor.risk_attribution()
        
        assert isinstance(risk_attr, RiskAttribution)
        
        # Check all attributes
        assert hasattr(risk_attr, 'portfolio_risk')
        assert hasattr(risk_attr, 'asset_risk_contributions')
        assert hasattr(risk_attr, 'marginal_risk_contributions')
        assert hasattr(risk_attr, 'component_var')
        assert hasattr(risk_attr, 'diversification_ratio')
        
        # Portfolio risk should be positive
        assert risk_attr.portfolio_risk > 0
        
        # Diversification ratio should be >= 1
        assert risk_attr.diversification_ratio >= 1.0
        
        # Risk contributions should sum to portfolio risk
        total_risk_contrib = sum(risk_attr.asset_risk_contributions.values())
        assert abs(total_risk_contrib - risk_attr.portfolio_risk) < 0.01
    
    def test_component_var(self, attributor):
        """Test component VaR calculation"""
        risk_attr = attributor.risk_attribution()
        
        # Component VaR should sum to approximately portfolio risk
        total_component_var = sum(risk_attr.component_var.values())
        assert abs(total_component_var - risk_attr.portfolio_risk) < 0.01
    
    def test_rolling_attribution(self, attributor):
        """Test rolling attribution analysis"""
        rolling_attr = attributor.rolling_attribution(window=252, step=60)
        
        assert isinstance(rolling_attr, pd.DataFrame)
        assert len(rolling_attr) > 0
        
        # Check required columns
        required_cols = [
            'Total Return', 'Benchmark Return', 'Active Return',
            'Allocation Effect', 'Selection Effect', 'Interaction Effect'
        ]
        for col in required_cols:
            assert col in rolling_attr.columns
    
    def test_sharpe_ratio(self, attributor):
        """Test Sharpe ratio calculation"""
        sharpe = attributor.calculate_sharpe_ratio(risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        assert np.isfinite(sharpe)
        # Reasonable bounds for Sharpe ratio
        assert -5 < sharpe < 10
    
    def test_information_ratio(self, attributor):
        """Test Information ratio calculation"""
        ir = attributor.calculate_information_ratio()
        
        assert isinstance(ir, float)
        assert np.isfinite(ir)
    
    def test_sortino_ratio(self, attributor):
        """Test Sortino ratio calculation"""
        sortino = attributor.calculate_sortino_ratio(risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        assert np.isfinite(sortino)
        assert -5 < sortino < 10
    
    def test_report_generation(self, attributor):
        """Test attribution report generation"""
        report = attributor.generate_attribution_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'PERFORMANCE ATTRIBUTION REPORT' in report
        assert 'Portfolio Return' in report
        assert 'Brinson Attribution' in report
        assert 'Risk Attribution' in report
    
    def test_weight_normalization(self, sample_returns):
        """Test that weights are properly normalized"""
        # Unnormalized weights
        portfolio_weights = {'Asset_A': 40, 'Asset_B': 30, 'Asset_C': 30}
        attr = PerformanceAttributor(sample_returns, portfolio_weights)
        
        # Should be normalized to 1
        assert abs(sum(attr.portfolio_weights.values()) - 1.0) < 1e-10
        
        # Proportions should be maintained
        assert abs(attr.portfolio_weights['Asset_A'] - 0.4) < 1e-10
    
    def test_date_range_filtering(self, attributor):
        """Test attribution with date range filtering"""
        # Full period
        attr_full = attributor.brinson_attribution()
        
        # Partial period
        attr_partial = attributor.brinson_attribution(
            start_date='2021-01-01',
            end_date='2022-12-31'
        )
        
        assert isinstance(attr_full, AttributionResult)
        assert isinstance(attr_partial, AttributionResult)
        
        # Returns should be different
        assert attr_full.total_return != attr_partial.total_return


class TestAttributionResult:
    """Test AttributionResult dataclass"""
    
    def test_attribution_result_creation(self):
        """Test AttributionResult creation"""
        result = AttributionResult(
            total_return=0.15,
            benchmark_return=0.12,
            active_return=0.03,
            allocation_effect=0.01,
            selection_effect=0.015,
            interaction_effect=0.005,
            asset_contributions={'Asset_A': 0.08, 'Asset_B': 0.07}
        )
        
        assert result.total_return == 0.15
        assert result.active_return == 0.03
        assert len(result.asset_contributions) == 2


class TestRiskAttribution:
    """Test RiskAttribution dataclass"""
    
    def test_risk_attribution_creation(self):
        """Test RiskAttribution creation"""
        result = RiskAttribution(
            portfolio_risk=0.15,
            asset_risk_contributions={'Asset_A': 0.08, 'Asset_B': 0.07},
            marginal_risk_contributions={'Asset_A': 0.16, 'Asset_B': 0.14},
            component_var={'Asset_A': 0.08, 'Asset_B': 0.07},
            diversification_ratio=1.2
        )
        
        assert result.portfolio_risk == 0.15
        assert result.diversification_ratio == 1.2
        assert len(result.asset_risk_contributions) == 2


def test_integration_workflow():
    """Test complete attribution workflow"""
    # Generate data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    returns = pd.DataFrame({
        'BTC': np.random.normal(0.001, 0.04, len(dates)),
        'ETH': np.random.normal(0.0012, 0.045, len(dates)),
        'SOL': np.random.normal(0.0015, 0.055, len(dates)),
    }, index=dates)
    
    # Initialize attributor
    portfolio_weights = {'BTC': 0.5, 'ETH': 0.3, 'SOL': 0.2}
    attr = PerformanceAttributor(returns, portfolio_weights)
    
    # Brinson attribution
    attribution = attr.brinson_attribution()
    assert attribution is not None
    assert abs(sum(attribution.asset_contributions.values()) - attribution.total_return) < 1e-6
    
    # Risk attribution
    risk_attr = attr.risk_attribution()
    assert risk_attr is not None
    assert risk_attr.portfolio_risk > 0
    
    # Factor attribution
    factor_contrib = attr.factor_attribution()
    assert factor_contrib is not None
    
    # Risk-adjusted metrics
    sharpe = attr.calculate_sharpe_ratio()
    ir = attr.calculate_information_ratio()
    sortino = attr.calculate_sortino_ratio()
    
    assert all(np.isfinite([sharpe, ir, sortino]))
    
    # Generate report
    report = attr.generate_attribution_report()
    assert 'PERFORMANCE ATTRIBUTION REPORT' in report


def test_edge_cases():
    """Test edge cases"""
    # Single asset portfolio
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    returns = pd.DataFrame({
        'Asset': np.random.normal(0, 0.02, len(dates))
    }, index=dates)
    
    attr = PerformanceAttributor(returns, {'Asset': 1.0})
    attribution = attr.brinson_attribution()
    
    assert attribution.total_return == attribution.benchmark_return
    assert abs(attribution.active_return) < 1e-10
    
    # Zero returns
    zero_returns = pd.DataFrame({
        'Asset_A': np.zeros(len(dates)),
        'Asset_B': np.zeros(len(dates))
    }, index=dates)
    
    attr_zero = PerformanceAttributor(
        zero_returns,
        {'Asset_A': 0.5, 'Asset_B': 0.5}
    )
    attribution_zero = attr_zero.brinson_attribution()
    
    assert attribution_zero.total_return == 0.0
    assert attribution_zero.benchmark_return == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
