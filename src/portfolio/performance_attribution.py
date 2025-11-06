"""
Performance Attribution Module
Phase 5 - Session 20: Performance Attribution Analysis

Provides:
- Brinson attribution (allocation + selection effects)
- Factor attribution analysis
- Risk attribution (risk contribution by asset)
- Time-weighted returns (TWR)
- Money-weighted returns (IRR)
- Rolling attribution analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import newton

logger = logging.getLogger(__name__)


@dataclass
class BrinsonAttribution:
    """Container for Brinson attribution results"""
    total_return: float
    benchmark_return: float
    active_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    asset_contributions: Dict[str, Dict[str, float]]


@dataclass
class RiskAttribution:
    """Container for risk attribution results"""
    portfolio_risk: float
    marginal_risk: Dict[str, float]
    component_risk: Dict[str, float]
    risk_contribution_pct: Dict[str, float]
    diversification_ratio: float


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float


class PerformanceAttributor:
    """
    Performance Attribution Analysis

    Provides comprehensive attribution analysis:
    - Brinson attribution (allocation vs selection)
    - Factor attribution
    - Risk attribution
    - Time/Money weighted returns
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        benchmark_weights: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Performance Attributor

        Args:
            returns: DataFrame of asset returns (index=dates, columns=symbols)
            weights: Dictionary of portfolio weights {symbol: weight}
            benchmark_weights: Benchmark weights (if None, uses equal weight)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.weights = pd.Series(weights)
        self.risk_free_rate = risk_free_rate

        # Align weights with returns columns
        self.weights = self.weights.reindex(returns.columns, fill_value=0)

        # Set benchmark weights (equal weight if not provided)
        if benchmark_weights is None:
            n_assets = len(returns.columns)
            self.benchmark_weights = pd.Series(
                {col: 1.0 / n_assets for col in returns.columns}
            )
        else:
            self.benchmark_weights = pd.Series(benchmark_weights)
            self.benchmark_weights = self.benchmark_weights.reindex(
                returns.columns, fill_value=0
            )

        # Calculate portfolio and benchmark returns
        self.portfolio_returns = (returns * self.weights).sum(axis=1)
        self.benchmark_returns = (returns * self.benchmark_weights).sum(axis=1)

    def brinson_attribution(self) -> BrinsonAttribution:
        """
        Brinson-Fachler Attribution Analysis

        Decomposes active return into:
        - Allocation Effect: Returns from overweight/underweight decisions
        - Selection Effect: Returns from asset picking within sectors
        - Interaction Effect: Combined effect of allocation and selection

        Returns:
            BrinsonAttribution dataclass
        """
        # Calculate returns
        portfolio_return = (1 + self.portfolio_returns).prod() - 1
        benchmark_return = (1 + self.benchmark_returns).prod() - 1
        active_return = portfolio_return - benchmark_return

        # Asset returns (buy and hold)
        asset_returns = (1 + self.returns).prod() - 1

        # Weight differences
        weight_diff = self.weights - self.benchmark_weights

        # Allocation Effect: (w_p - w_b) * (R_b - R_benchmark)
        allocation_effects = {}
        selection_effects = {}
        interaction_effects = {}

        for asset in self.returns.columns:
            w_diff = weight_diff[asset]
            r_asset = asset_returns[asset]
            r_benchmark = benchmark_return

            # Allocation: weight difference * (asset return - benchmark return)
            allocation = w_diff * (r_asset - r_benchmark)

            # Selection: benchmark weight * (asset return - benchmark return)
            # But we use portfolio weight for actual selection
            selection = self.benchmark_weights[asset] * (r_asset - r_benchmark)

            # Interaction: (w_p - w_b) * (R_asset - R_benchmark)
            interaction = w_diff * (r_asset - r_benchmark)

            allocation_effects[asset] = allocation
            selection_effects[asset] = selection
            interaction_effects[asset] = interaction

        # Total effects
        total_allocation = sum(allocation_effects.values())
        total_selection = sum(selection_effects.values())
        total_interaction = sum(interaction_effects.values())

        # Asset contributions
        asset_contributions = {}
        for asset in self.returns.columns:
            asset_contributions[asset] = {
                'allocation': allocation_effects[asset],
                'selection': selection_effects[asset],
                'interaction': interaction_effects[asset],
                'total': (allocation_effects[asset] +
                         selection_effects[asset] +
                         interaction_effects[asset])
            }

        return BrinsonAttribution(
            total_return=portfolio_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            asset_contributions=asset_contributions
        )

    def risk_attribution(self) -> RiskAttribution:
        """
        Risk Attribution Analysis

        Calculates each asset's contribution to portfolio risk

        Returns:
            RiskAttribution dataclass
        """
        # Portfolio variance
        cov_matrix = self.returns.cov()
        portfolio_variance = self.weights.T @ cov_matrix @ self.weights
        portfolio_risk = np.sqrt(portfolio_variance * 252)  # Annualized

        # Marginal risk contribution
        marginal_risk = {}
        for asset in self.returns.columns:
            # Marginal contribution to variance
            marginal_var = 2 * (cov_matrix[asset] @ self.weights)
            marginal_risk[asset] = marginal_var / (2 * np.sqrt(portfolio_variance))

        # Component risk = Marginal risk × Weight
        component_risk = {}
        for asset in self.returns.columns:
            component_risk[asset] = marginal_risk[asset] * self.weights[asset] * np.sqrt(252)

        # Risk contribution percentage
        total_component_risk = sum(abs(v) for v in component_risk.values())
        risk_contribution_pct = {}
        for asset in self.returns.columns:
            if total_component_risk > 0:
                risk_contribution_pct[asset] = component_risk[asset] / total_component_risk
            else:
                risk_contribution_pct[asset] = 0

        # Diversification ratio
        # Weighted average of individual volatilities / Portfolio volatility
        individual_vols = self.returns.std() * np.sqrt(252)
        weighted_vol = (self.weights * individual_vols).sum()
        diversification_ratio = weighted_vol / portfolio_risk if portfolio_risk > 0 else 1

        return RiskAttribution(
            portfolio_risk=portfolio_risk,
            marginal_risk=marginal_risk,
            component_risk=component_risk,
            risk_contribution_pct=risk_contribution_pct,
            diversification_ratio=diversification_ratio
        )

    def time_weighted_return(self) -> float:
        """
        Calculate Time-Weighted Return (TWR)

        TWR is the compound growth rate, independent of cash flows

        Returns:
            Time-weighted return (annualized)
        """
        # Compound return
        total_return = (1 + self.portfolio_returns).prod() - 1

        # Annualize
        n_days = len(self.portfolio_returns)
        n_years = n_days / 252
        annualized = (1 + total_return) ** (1 / n_years) - 1

        return annualized

    def money_weighted_return(
        self,
        cash_flows: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate Money-Weighted Return (MWR) / Internal Rate of Return (IRR)

        MWR accounts for timing and size of cash flows

        Args:
            cash_flows: Series of cash flows (positive=inflow, negative=outflow)
                       If None, assumes no intermediate cash flows

        Returns:
            Money-weighted return (annualized IRR)
        """
        if cash_flows is None:
            # No cash flows, TWR = MWR
            return self.time_weighted_return()

        # Calculate IRR using Newton's method
        # NPV = sum(CF_t / (1+r)^t) = 0

        def npv(rate):
            pv = 0
            for i, (date, cf) in enumerate(cash_flows.items()):
                days_elapsed = (date - cash_flows.index[0]).days
                years = days_elapsed / 365
                pv += cf / ((1 + rate) ** years)
            return pv

        try:
            irr = newton(npv, 0.1, maxiter=100)
            return irr
        except (RuntimeError, ValueError, ZeroDivisionError, AttributeError) as e:
            # Fallback to TWR if IRR calculation fails (non-convergence, invalid data, etc.)
            logger.warning(f"IRR calculation failed: {e}. Falling back to TWR.")
            return self.time_weighted_return()

    def rolling_attribution(
        self,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling attribution over time

        Args:
            window: Rolling window size in days

        Returns:
            DataFrame with rolling attribution metrics
        """
        results = []

        for i in range(window, len(self.returns)):
            window_returns = self.returns.iloc[i-window:i]

            # Create temporary attributor for window
            temp_attributor = PerformanceAttributor(
                window_returns,
                self.weights.to_dict(),
                self.benchmark_weights.to_dict(),
                self.risk_free_rate
            )

            attribution = temp_attributor.brinson_attribution()

            results.append({
                'date': self.returns.index[i],
                'portfolio_return': attribution.total_return,
                'benchmark_return': attribution.benchmark_return,
                'active_return': attribution.active_return,
                'allocation_effect': attribution.allocation_effect,
                'selection_effect': attribution.selection_effect
            })

        return pd.DataFrame(results).set_index('date')

    def factor_attribution(
        self,
        factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Factor Attribution Analysis

        Attributes returns to common factors (market, value, momentum, etc.)

        Args:
            factors: DataFrame of factor returns (if None, uses simple factors)

        Returns:
            Dictionary of factor contributions
        """
        if factors is None:
            # Create simple factors from returns
            factors = pd.DataFrame({
                'Market': self.returns.mean(axis=1),  # Market factor
                'Value': self.returns.iloc[:, 0] - self.returns.iloc[:, -1],  # Value spread
                'Momentum': self.returns.rolling(21).mean().mean(axis=1)  # Momentum
            }, index=self.returns.index)

        # Regression: portfolio_returns = alpha + beta * factors + epsilon
        from sklearn.linear_model import LinearRegression

        # Align data
        common_index = self.portfolio_returns.index.intersection(factors.index)
        y = self.portfolio_returns.loc[common_index].values.reshape(-1, 1)
        X = factors.loc[common_index].values

        # Fit regression
        model = LinearRegression()
        model.fit(X, y)

        # Calculate factor contributions
        factor_contributions = {}
        for i, factor_name in enumerate(factors.columns):
            beta = model.coef_[0][i]
            factor_return = factors.iloc[:, i].mean() * 252  # Annualized
            contribution = beta * factor_return
            factor_contributions[factor_name] = contribution

        # Alpha (unexplained return)
        factor_contributions['Alpha'] = model.intercept_[0] * 252

        return factor_contributions

    def calculate_performance_metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            benchmark_returns: Benchmark returns for information ratio

        Returns:
            PerformanceMetrics dataclass
        """
        # Total return
        total_return = (1 + self.portfolio_returns).prod() - 1

        # Annualized return
        n_days = len(self.portfolio_returns)
        n_years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1

        # Volatility (annualized)
        volatility = self.portfolio_returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

        # Omega Ratio (probability weighted ratio of gains to losses)
        threshold = 0
        gains = self.portfolio_returns[self.portfolio_returns > threshold] - threshold
        losses = threshold - self.portfolio_returns[self.portfolio_returns < threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else 0

        # Information Ratio
        if benchmark_returns is not None:
            active_returns = self.portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            information_ratio=information_ratio
        )

    def asset_contribution_to_return(self) -> Dict[str, float]:
        """
        Calculate each asset's contribution to total portfolio return

        Returns:
            Dictionary of asset contributions to return
        """
        # Total return per asset (weighted)
        asset_total_returns = (1 + self.returns).prod() - 1
        contributions = {}

        for asset in self.returns.columns:
            # Contribution = weight × asset return
            contributions[asset] = self.weights[asset] * asset_total_returns[asset]

        return contributions

    def generate_attribution_summary(self) -> Dict:
        """
        Generate comprehensive attribution summary

        Returns:
            Dictionary with all attribution analysis
        """
        brinson = self.brinson_attribution()
        risk_attr = self.risk_attribution()
        metrics = self.calculate_performance_metrics()
        asset_contributions = self.asset_contribution_to_return()
        factor_attr = self.factor_attribution()

        return {
            'brinson_attribution': brinson,
            'risk_attribution': risk_attr,
            'performance_metrics': metrics,
            'asset_contributions': asset_contributions,
            'factor_attribution': factor_attr,
            'time_weighted_return': self.time_weighted_return(),
            'money_weighted_return': self.money_weighted_return()
        }


def generate_attribution_report(attributor: PerformanceAttributor) -> str:
    """
    Generate text report of attribution analysis

    Args:
        attributor: PerformanceAttributor instance

    Returns:
        Formatted attribution report string
    """
    summary = attributor.generate_attribution_summary()
    brinson = summary['brinson_attribution']
    risk_attr = summary['risk_attribution']
    metrics = summary['performance_metrics']

    report = []
    report.append("=" * 60)
    report.append("PERFORMANCE ATTRIBUTION REPORT")
    report.append("=" * 60)
    report.append("")

    report.append("PERFORMANCE METRICS")
    report.append("-" * 60)
    report.append(f"Total Return:         {metrics.total_return:.2%}")
    report.append(f"Annualized Return:    {metrics.annualized_return:.2%}")
    report.append(f"Volatility:           {metrics.volatility:.2%}")
    report.append(f"Sharpe Ratio:         {metrics.sharpe_ratio:.3f}")
    report.append(f"Sortino Ratio:        {metrics.sortino_ratio:.3f}")
    report.append(f"Calmar Ratio:         {metrics.calmar_ratio:.3f}")
    report.append(f"Max Drawdown:         {metrics.max_drawdown:.2%}")
    report.append("")

    report.append("BRINSON ATTRIBUTION")
    report.append("-" * 60)
    report.append(f"Portfolio Return:     {brinson.total_return:.2%}")
    report.append(f"Benchmark Return:     {brinson.benchmark_return:.2%}")
    report.append(f"Active Return:        {brinson.active_return:.2%}")
    report.append("")
    report.append("Attribution Effects:")
    report.append(f"  Allocation Effect:  {brinson.allocation_effect:.2%}")
    report.append(f"  Selection Effect:   {brinson.selection_effect:.2%}")
    report.append(f"  Interaction Effect: {brinson.interaction_effect:.2%}")
    report.append("")

    report.append("ASSET CONTRIBUTIONS")
    report.append("-" * 60)
    for asset, contrib in brinson.asset_contributions.items():
        report.append(f"{asset}:")
        report.append(f"  Allocation:  {contrib['allocation']:+.2%}")
        report.append(f"  Selection:   {contrib['selection']:+.2%}")
        report.append(f"  Total:       {contrib['total']:+.2%}")
    report.append("")

    report.append("RISK ATTRIBUTION")
    report.append("-" * 60)
    report.append(f"Portfolio Risk:        {risk_attr.portfolio_risk:.2%}")
    report.append(f"Diversification Ratio: {risk_attr.diversification_ratio:.3f}")
    report.append("")
    report.append("Risk Contribution (%):")
    for asset, contrib_pct in risk_attr.risk_contribution_pct.items():
        report.append(f"  {asset}: {contrib_pct:.2%}")
    report.append("")

    report.append("FACTOR ATTRIBUTION")
    report.append("-" * 60)
    for factor, contribution in summary['factor_attribution'].items():
        report.append(f"{factor}: {contribution:+.2%}")
    report.append("")

    report.append("=" * 60)

    return "\n".join(report)
