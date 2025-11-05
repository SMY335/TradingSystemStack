"""
Performance Attribution Module

Implements comprehensive performance attribution analysis including:
- Factor attribution (market, value, momentum, volatility)
- Strategy attribution (contribution by asset)
- Brinson attribution (allocation + selection effects)
- Time-weighted returns (TWR)
- Money-weighted returns (MWR/IRR)
- Risk-adjusted performance metrics
- Risk attribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import newton
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AttributionResult:
    """Container for attribution analysis results"""
    total_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    asset_contributions: Dict[str, float]


@dataclass
class RiskAttribution:
    """Container for risk attribution results"""
    portfolio_risk: float
    asset_risk_contributions: Dict[str, float]
    marginal_risk_contributions: Dict[str, float]
    component_var: Dict[str, float]
    diversification_ratio: float


class PerformanceAttributor:
    """
    Comprehensive Performance Attribution Analyzer
    
    Features:
    - Brinson attribution (allocation, selection, interaction)
    - Factor-based attribution
    - Asset contribution analysis
    - Time-weighted and money-weighted returns
    - Risk attribution
    - Rolling attribution analysis
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]] = None,
        prices: Optional[pd.DataFrame] = None
    ):
        """
        Initialize Performance Attributor
        
        Args:
            returns: DataFrame of asset returns (columns = assets, index = dates)
            portfolio_weights: Dictionary of portfolio weights
            benchmark_weights: Dictionary of benchmark weights (if None, equal weights)
            prices: DataFrame of asset prices (optional, for TWR/MWR)
        """
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Normalize portfolio weights
        total_weight = sum(portfolio_weights.values())
        self.portfolio_weights = {k: v / total_weight for k, v in portfolio_weights.items()}
        
        # Set benchmark weights
        if benchmark_weights is None:
            self.benchmark_weights = {asset: 1.0 / self.n_assets for asset in self.assets}
        else:
            total_bench = sum(benchmark_weights.values())
            self.benchmark_weights = {k: v / total_bench for k, v in benchmark_weights.items()}
        
        self.prices = prices
        
        # Calculate portfolio and benchmark returns
        self.portfolio_returns = self._calculate_weighted_returns(self.portfolio_weights)
        self.benchmark_returns = self._calculate_weighted_returns(self.benchmark_weights)
    
    def _calculate_weighted_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted portfolio returns"""
        weights_array = np.array([weights[asset] for asset in self.assets])
        return (self.returns * weights_array).sum(axis=1)
    
    def calculate_twr(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> float:
        """
        Calculate Time-Weighted Return
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Time-weighted return
        """
        returns = self.portfolio_returns.copy()
        
        if start_date:
            returns = returns[returns.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]
        
        # TWR = Product of (1 + r_i) - 1
        twr = (1 + returns).prod() - 1
        return twr
    
    def calculate_mwr(
        self,
        cash_flows: pd.Series,
        start_value: float,
        end_value: float
    ) -> float:
        """
        Calculate Money-Weighted Return (Internal Rate of Return)
        
        Args:
            cash_flows: Series of cash flows indexed by date
            start_value: Initial portfolio value
            end_value: Final portfolio value
            
        Returns:
            Money-weighted return (IRR)
        """
        # IRR calculation using Newton's method
        def npv(rate):
            pv = start_value
            for date, cf in cash_flows.items():
                days = (date - cash_flows.index[0]).days
                pv += cf / (1 + rate) ** (days / 365.25)
            pv -= end_value / (1 + rate) ** ((cash_flows.index[-1] - cash_flows.index[0]).days / 365.25)
            return pv
        
        try:
            mwr = newton(npv, 0.1)  # Start with 10% guess
            return mwr
        except:
            return np.nan
    
    def brinson_attribution(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> AttributionResult:
        """
        Perform Brinson-Fachler attribution analysis
        
        Decomposes active return into:
        - Allocation effect: from over/underweighting sectors
        - Selection effect: from picking better/worse performing assets
        - Interaction effect: combined allocation and selection
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            AttributionResult object
        """
        # Filter returns by date range
        returns = self.returns.copy()
        if start_date:
            returns = returns[returns.index >= start_date]
        if end_date:
            returns = returns[returns.index <= end_date]
        
        # Calculate period returns for each asset
        asset_returns = {}
        for asset in self.assets:
            asset_returns[asset] = (1 + returns[asset]).prod() - 1
        
        # Calculate portfolio and benchmark returns
        portfolio_return = sum(
            self.portfolio_weights[asset] * asset_returns[asset]
            for asset in self.assets
        )
        
        benchmark_return = sum(
            self.benchmark_weights[asset] * asset_returns[asset]
            for asset in self.assets
        )
        
        # Attribution effects
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        asset_contributions = {}
        
        for asset in self.assets:
            w_p = self.portfolio_weights[asset]
            w_b = self.benchmark_weights[asset]
            r_a = asset_returns[asset]
            
            # Allocation effect: (w_p - w_b) * (r_b - r_benchmark)
            # Simplified: (w_p - w_b) * r_b for each asset
            allocation = (w_p - w_b) * benchmark_return
            
            # Selection effect: w_b * (r_a - r_benchmark)
            selection = w_b * (r_a - benchmark_return)
            
            # Interaction effect: (w_p - w_b) * (r_a - r_benchmark)
            interaction = (w_p - w_b) * (r_a - benchmark_return)
            
            allocation_effect += allocation
            selection_effect += selection
            interaction_effect += interaction
            
            # Asset contribution to total return
            asset_contributions[asset] = w_p * r_a
        
        active_return = portfolio_return - benchmark_return
        
        return AttributionResult(
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            asset_contributions=asset_contributions
        )
    
    def factor_attribution(
        self,
        factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Perform factor-based attribution
        
        Args:
            factors: DataFrame of factor returns (if None, creates synthetic factors)
            
        Returns:
            Dictionary of factor contributions
        """
        if factors is None:
            # Create synthetic factors from returns
            factors = self._create_synthetic_factors()
        
        # Align dates
        common_dates = self.portfolio_returns.index.intersection(factors.index)
        port_returns = self.portfolio_returns.loc[common_dates]
        factor_returns = factors.loc[common_dates]
        
        # Run regression: portfolio returns = alpha + beta * factors
        X = factor_returns.values
        y = port_returns.values
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
        alpha = betas[0]
        factor_betas = betas[1:]
        
        # Calculate factor contributions
        factor_contributions = {}
        factor_contributions['Alpha'] = alpha * len(common_dates)
        
        for i, factor_name in enumerate(factor_returns.columns):
            # Factor contribution = beta * factor_return
            factor_contrib = factor_betas[i] * factor_returns[factor_name].sum()
            factor_contributions[factor_name] = factor_contrib
        
        return factor_contributions
    
    def _create_synthetic_factors(self) -> pd.DataFrame:
        """Create synthetic market factors from returns"""
        factors = pd.DataFrame(index=self.returns.index)
        
        # Market factor (equal-weighted average)
        factors['Market'] = self.returns.mean(axis=1)
        
        # Momentum factor (12-1 month momentum)
        if len(self.returns) > 252:
            mom_start = self.returns.rolling(252).mean()
            mom_recent = self.returns.rolling(21).mean()
            factors['Momentum'] = (mom_recent - mom_start).mean(axis=1)
        else:
            factors['Momentum'] = 0
        
        # Volatility factor
        factors['Volatility'] = self.returns.rolling(21).std().mean(axis=1)
        
        return factors.fillna(0)
    
    def calculate_asset_contributions(
        self,
        period: str = 'full'
    ) -> pd.DataFrame:
        """
        Calculate contribution of each asset to portfolio return
        
        Args:
            period: 'full', 'daily', 'monthly', 'yearly'
            
        Returns:
            DataFrame with asset contributions
        """
        if period == 'full':
            contributions = {}
            for asset in self.assets:
                total_return = (1 + self.returns[asset]).prod() - 1
                contribution = self.portfolio_weights[asset] * total_return
                contributions[asset] = contribution
            
            return pd.DataFrame([contributions], index=['Full Period'])
        
        elif period == 'daily':
            contributions = self.returns * np.array([
                self.portfolio_weights[asset] for asset in self.assets
            ])
            return contributions
        
        elif period == 'monthly':
            monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            contributions = monthly_returns * np.array([
                self.portfolio_weights[asset] for asset in self.assets
            ])
            return contributions
        
        elif period == 'yearly':
            yearly_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
            contributions = yearly_returns * np.array([
                self.portfolio_weights[asset] for asset in self.assets
            ])
            return contributions
        
        else:
            raise ValueError(f"Unknown period: {period}")
    
    def risk_attribution(self) -> RiskAttribution:
        """
        Perform risk attribution analysis
        
        Returns:
            RiskAttribution object with detailed risk decomposition
        """
        # Calculate covariance matrix
        cov_matrix = self.returns.cov()
        
        # Portfolio weights as array
        weights = np.array([self.portfolio_weights[asset] for asset in self.assets])
        
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_risk = np.sqrt(portfolio_variance * 252)  # Annualized
        
        # Marginal risk contribution
        marginal_risk = cov_matrix @ weights
        marginal_risk_contrib = {}
        for i, asset in enumerate(self.assets):
            marginal_risk_contrib[asset] = marginal_risk[i] * np.sqrt(252)
        
        # Component risk contribution (absolute)
        component_risk = weights * marginal_risk
        asset_risk_contributions = {}
        for i, asset in enumerate(self.assets):
            asset_risk_contributions[asset] = component_risk[i] * np.sqrt(252)
        
        # Component VaR (simplified)
        component_var = {}
        for i, asset in enumerate(self.assets):
            # Proportional VaR contribution
            component_var[asset] = (component_risk[i] / portfolio_variance) * portfolio_risk
        
        # Diversification ratio
        weighted_vol = sum(
            self.portfolio_weights[asset] * self.returns[asset].std() * np.sqrt(252)
            for asset in self.assets
        )
        diversification_ratio = weighted_vol / portfolio_risk if portfolio_risk > 0 else 1.0
        
        return RiskAttribution(
            portfolio_risk=portfolio_risk,
            asset_risk_contributions=asset_risk_contributions,
            marginal_risk_contributions=marginal_risk_contrib,
            component_var=component_var,
            diversification_ratio=diversification_ratio
        )
    
    def rolling_attribution(
        self,
        window: int = 252,
        step: int = 21
    ) -> pd.DataFrame:
        """
        Calculate rolling attribution over time
        
        Args:
            window: Rolling window size in days
            step: Step size between windows
            
        Returns:
            DataFrame with rolling attribution results
        """
        results = []
        dates = []
        
        for i in range(window, len(self.returns), step):
            start_idx = i - window
            end_idx = i
            
            # Get window data
            window_returns = self.returns.iloc[start_idx:end_idx]
            
            # Create temporary attributor
            temp_attributor = PerformanceAttributor(
                window_returns,
                self.portfolio_weights,
                self.benchmark_weights
            )
            
            # Calculate attribution
            attr = temp_attributor.brinson_attribution()
            
            results.append({
                'Total Return': attr.total_return,
                'Benchmark Return': attr.benchmark_return,
                'Active Return': attr.active_return,
                'Allocation Effect': attr.allocation_effect,
                'Selection Effect': attr.selection_effect,
                'Interaction Effect': attr.interaction_effect
            })
            
            dates.append(self.returns.index[end_idx - 1])
        
        return pd.DataFrame(results, index=dates)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = self.portfolio_returns.mean() * 252 - risk_free_rate
        volatility = self.portfolio_returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0
    
    def calculate_information_ratio(self) -> float:
        """Calculate Information Ratio (active return / tracking error)"""
        active_returns = self.portfolio_returns - self.benchmark_returns
        active_return = active_returns.mean() * 252
        tracking_error = active_returns.std() * np.sqrt(252)
        return active_return / tracking_error if tracking_error > 0 else 0
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        excess_returns = self.portfolio_returns.mean() * 252 - risk_free_rate
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_std if downside_std > 0 else 0
    
    def generate_attribution_report(self) -> str:
        """
        Generate comprehensive attribution report
        
        Returns:
            Formatted attribution report string
        """
        attribution = self.brinson_attribution()
        risk_attr = self.risk_attribution()
        
        report = ["=" * 80]
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Performance summary
        report.append("Performance Summary:")
        report.append(f"  Portfolio Return: {attribution.total_return:.4f} ({attribution.total_return*100:.2f}%)")
        report.append(f"  Benchmark Return: {attribution.benchmark_return:.4f} ({attribution.benchmark_return*100:.2f}%)")
        report.append(f"  Active Return: {attribution.active_return:.4f} ({attribution.active_return*100:.2f}%)")
        report.append("")
        
        # Brinson attribution
        report.append("Brinson Attribution:")
        report.append(f"  Allocation Effect: {attribution.allocation_effect:.4f} ({attribution.allocation_effect*100:.2f}%)")
        report.append(f"  Selection Effect: {attribution.selection_effect:.4f} ({attribution.selection_effect*100:.2f}%)")
        report.append(f"  Interaction Effect: {attribution.interaction_effect:.4f} ({attribution.interaction_effect*100:.2f}%)")
        report.append("")
        
        # Asset contributions
        report.append("Asset Contributions to Return:")
        sorted_contributions = sorted(
            attribution.asset_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for asset, contrib in sorted_contributions:
            report.append(f"  {asset}: {contrib:.4f} ({contrib*100:.2f}%)")
        report.append("")
        
        # Risk attribution
        report.append("Risk Attribution:")
        report.append(f"  Portfolio Risk (ann.): {risk_attr.portfolio_risk:.4f} ({risk_attr.portfolio_risk*100:.2f}%)")
        report.append(f"  Diversification Ratio: {risk_attr.diversification_ratio:.4f}")
        report.append("")
        
        report.append("Risk Contributions by Asset:")
        sorted_risk = sorted(
            risk_attr.asset_risk_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for asset, risk_contrib in sorted_risk:
            report.append(f"  {asset}: {risk_contrib:.4f} ({risk_contrib*100:.2f}%)")
        report.append("")
        
        # Risk-adjusted metrics
        report.append("Risk-Adjusted Performance:")
        report.append(f"  Sharpe Ratio: {self.calculate_sharpe_ratio():.4f}")
        report.append(f"  Information Ratio: {self.calculate_information_ratio():.4f}")
        report.append(f"  Sortino Ratio: {self.calculate_sortino_ratio():.4f}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def example_usage():
    """Example usage of PerformanceAttributor"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    n_assets = 5
    
    returns_data = {}
    for i in range(n_assets):
        mu = 0.0005 * (i + 1)
        sigma = 0.02 * (i + 1)
        returns_data[f'Asset_{i+1}'] = np.random.normal(mu, sigma, len(dates))
    
    returns = pd.DataFrame(returns_data, index=dates)
    
    # Portfolio weights
    portfolio_weights = {f'Asset_{i+1}': [0.3, 0.25, 0.2, 0.15, 0.1][i] for i in range(n_assets)}
    
    # Initialize attributor
    attributor = PerformanceAttributor(returns, portfolio_weights)
    
    # Generate report
    print(attributor.generate_attribution_report())


if __name__ == "__main__":
    example_usage()
