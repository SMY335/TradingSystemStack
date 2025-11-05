"""
Risk Management Module for Portfolio Analysis
Phase 5 - Session 19: Advanced Risk Management

Provides:
- VaR (Value at Risk) calculations: historical, parametric, Monte Carlo
- CVaR (Conditional VaR / Expected Shortfall)
- Stress testing scenarios
- Monte Carlo simulations
- Correlation analysis
- Tail risk metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy import stats
from datetime import datetime


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    calmar_ratio: float


@dataclass
class StressTestResult:
    """Container for stress test results"""
    scenario_name: str
    portfolio_return: float
    portfolio_loss: float
    worst_asset: str
    worst_asset_loss: float
    diversification_benefit: float


class RiskManager:
    """
    Advanced Risk Management for Portfolio Analysis

    Features:
    - Multiple VaR calculation methods
    - CVaR (Expected Shortfall)
    - Stress testing
    - Monte Carlo simulation
    - Correlation breakdown analysis
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float = 0.02,
        confidence_levels: List[float] = [0.95, 0.99]
    ):
        """
        Initialize Risk Manager

        Args:
            returns: DataFrame of asset returns (index=dates, columns=symbols)
            weights: Dictionary of portfolio weights {symbol: weight}
            risk_free_rate: Annual risk-free rate (default 2%)
            confidence_levels: Confidence levels for VaR/CVaR
        """
        self.returns = returns
        self.weights = pd.Series(weights)
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels

        # Calculate portfolio returns
        self.portfolio_returns = (returns * self.weights).sum(axis=1)

        # Align weights with returns columns
        self.weights = self.weights.reindex(returns.columns, fill_value=0)

    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics

        Returns:
            RiskMetrics dataclass with all metrics
        """
        # VaR and CVaR at different confidence levels
        var_95 = self.calculate_var(method='historical', confidence=0.95)
        var_99 = self.calculate_var(method='historical', confidence=0.99)
        cvar_95 = self.calculate_cvar(confidence=0.95)
        cvar_99 = self.calculate_cvar(confidence=0.99)

        # Volatility (annualized)
        volatility = self.portfolio_returns.std() * np.sqrt(252)

        # Sharpe Ratio
        annual_return = self.portfolio_returns.mean() * 252
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        max_drawdown = self.calculate_max_drawdown()

        # Tail risk metrics
        skewness = self.portfolio_returns.skew()
        kurtosis = self.portfolio_returns.kurtosis()

        # Calmar Ratio (return / max drawdown)
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            skewness=skewness,
            kurtosis=kurtosis,
            calmar_ratio=calmar_ratio
        )

    def calculate_var(
        self,
        method: str = 'historical',
        confidence: float = 0.95,
        n_scenarios: int = 10000
    ) -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            method: 'historical', 'parametric', or 'monte_carlo'
            confidence: Confidence level (e.g., 0.95 for 95%)
            n_scenarios: Number of scenarios for Monte Carlo

        Returns:
            VaR value (negative = loss)
        """
        if method == 'historical':
            return self.portfolio_returns.quantile(1 - confidence)

        elif method == 'parametric':
            mean = self.portfolio_returns.mean()
            std = self.portfolio_returns.std()
            z_score = stats.norm.ppf(1 - confidence)
            return mean + z_score * std

        elif method == 'monte_carlo':
            simulated_returns = self.monte_carlo_simulation(n_scenarios)
            return np.percentile(simulated_returns, (1 - confidence) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall

        CVaR is the expected loss given that VaR threshold is exceeded

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR value (negative = loss)
        """
        var = self.calculate_var(method='historical', confidence=confidence)
        # Returns worse than VaR
        tail_returns = self.portfolio_returns[self.portfolio_returns <= var]
        return tail_returns.mean() if len(tail_returns) > 0 else var

    def calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown

        Returns:
            Maximum drawdown as negative percentage
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def monte_carlo_simulation(
        self,
        n_scenarios: int = 10000,
        time_horizon: int = 1
    ) -> np.ndarray:
        """
        Run Monte Carlo simulation for portfolio returns

        Args:
            n_scenarios: Number of scenarios to simulate
            time_horizon: Time horizon in days

        Returns:
            Array of simulated portfolio returns
        """
        # Calculate mean and covariance
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values

        # Cholesky decomposition for correlated random variables
        L = np.linalg.cholesky(cov_matrix)

        # Generate random scenarios
        random_returns = np.random.normal(0, 1, (n_scenarios, len(mean_returns)))

        # Apply correlation structure
        correlated_returns = random_returns @ L.T

        # Add mean and scale by time horizon
        simulated_asset_returns = mean_returns + correlated_returns * np.sqrt(time_horizon)

        # Calculate portfolio returns
        portfolio_returns = simulated_asset_returns @ self.weights.values

        return portfolio_returns

    def stress_test(self) -> List[StressTestResult]:
        """
        Run stress tests on portfolio

        Tests various market scenarios:
        - Market crash (-20%, -50%)
        - Volatility spike (3x, 5x normal)
        - Correlation breakdown (all correlations → 1)

        Returns:
            List of StressTestResult for each scenario
        """
        results = []

        # Scenario 1: Market Crash -20%
        crash_20_returns = self.returns - 0.20
        crash_20_portfolio = (crash_20_returns * self.weights).sum(axis=1).mean()
        worst_asset = (crash_20_returns.mean() * self.weights).idxmin()
        results.append(StressTestResult(
            scenario_name="Market Crash -20%",
            portfolio_return=crash_20_portfolio,
            portfolio_loss=crash_20_portfolio * 100,
            worst_asset=worst_asset,
            worst_asset_loss=crash_20_returns[worst_asset].mean() * 100,
            diversification_benefit=(crash_20_portfolio - crash_20_returns.mean().min()) * 100
        ))

        # Scenario 2: Market Crash -50%
        crash_50_returns = self.returns - 0.50
        crash_50_portfolio = (crash_50_returns * self.weights).sum(axis=1).mean()
        worst_asset = (crash_50_returns.mean() * self.weights).idxmin()
        results.append(StressTestResult(
            scenario_name="Market Crash -50%",
            portfolio_return=crash_50_portfolio,
            portfolio_loss=crash_50_portfolio * 100,
            worst_asset=worst_asset,
            worst_asset_loss=crash_50_returns[worst_asset].mean() * 100,
            diversification_benefit=(crash_50_portfolio - crash_50_returns.mean().min()) * 100
        ))

        # Scenario 3: Volatility Spike 3x
        vol_spike = self.returns.std() * 3
        vol_spike_returns = np.random.normal(
            self.returns.mean(),
            vol_spike,
            size=self.returns.shape
        )
        vol_spike_returns = pd.DataFrame(
            vol_spike_returns,
            columns=self.returns.columns
        )
        vol_spike_portfolio = (vol_spike_returns * self.weights).sum(axis=1).mean()
        results.append(StressTestResult(
            scenario_name="Volatility Spike 3x",
            portfolio_return=vol_spike_portfolio,
            portfolio_loss=vol_spike_portfolio * 100,
            worst_asset=vol_spike_returns.mean().idxmin(),
            worst_asset_loss=vol_spike_returns.mean().min() * 100,
            diversification_benefit=0
        ))

        # Scenario 4: Correlation Breakdown (all → 1)
        # All assets move together perfectly
        avg_return = self.returns.mean().mean()
        avg_std = self.returns.std().mean()
        common_shock = np.random.normal(avg_return, avg_std, len(self.returns))
        corr_breakdown_returns = pd.DataFrame(
            {col: common_shock for col in self.returns.columns},
            index=self.returns.index
        )
        corr_breakdown_portfolio = (corr_breakdown_returns * self.weights).sum(axis=1).mean()
        results.append(StressTestResult(
            scenario_name="Correlation Breakdown",
            portfolio_return=corr_breakdown_portfolio,
            portfolio_loss=corr_breakdown_portfolio * 100,
            worst_asset="ALL (correlated)",
            worst_asset_loss=avg_return * 100,
            diversification_benefit=0  # No diversification when corr=1
        ))

        return results

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio assets

        Returns:
            Correlation matrix DataFrame
        """
        return self.returns.corr()

    def rolling_var(
        self,
        window: int = 30,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Calculate rolling VaR over time

        Args:
            window: Rolling window size in days
            confidence: Confidence level

        Returns:
            Series of rolling VaR values
        """
        return self.portfolio_returns.rolling(window).quantile(1 - confidence)

    def marginal_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Marginal VaR for each asset

        Marginal VaR = change in portfolio VaR from small change in position

        Returns:
            Dictionary of marginal VaR per asset
        """
        portfolio_var = self.calculate_var('historical', confidence)
        marginal_vars = {}

        # Small perturbation
        delta = 0.01

        for asset in self.returns.columns:
            # Increase weight by delta
            new_weights = self.weights.copy()
            new_weights[asset] += delta
            new_weights = new_weights / new_weights.sum()  # Renormalize

            # Calculate new portfolio returns
            new_portfolio_returns = (self.returns * new_weights).sum(axis=1)
            new_var = new_portfolio_returns.quantile(1 - confidence)

            # Marginal VaR
            marginal_vars[asset] = (new_var - portfolio_var) / delta

        return marginal_vars

    def component_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Component VaR for each asset

        Component VaR = Marginal VaR × Position size
        Sum of Component VaRs = Portfolio VaR

        Returns:
            Dictionary of component VaR per asset
        """
        marginal_vars = self.marginal_var(confidence)
        component_vars = {}

        for asset, marginal_var in marginal_vars.items():
            component_vars[asset] = marginal_var * self.weights[asset]

        return component_vars

    def get_risk_summary(self) -> Dict:
        """
        Get comprehensive risk summary

        Returns:
            Dictionary with all risk metrics and analysis
        """
        metrics = self.calculate_risk_metrics()
        stress_results = self.stress_test()
        correlation = self.correlation_analysis()
        marginal_var = self.marginal_var()
        component_var = self.component_var()

        return {
            'metrics': metrics,
            'stress_tests': stress_results,
            'correlation_matrix': correlation,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'portfolio_return_annual': self.portfolio_returns.mean() * 252,
            'portfolio_volatility_annual': self.portfolio_returns.std() * np.sqrt(252)
        }


def generate_risk_report(risk_manager: RiskManager) -> str:
    """
    Generate a text report of risk analysis

    Args:
        risk_manager: RiskManager instance

    Returns:
        Formatted risk report string
    """
    summary = risk_manager.get_risk_summary()
    metrics = summary['metrics']

    report = []
    report.append("=" * 60)
    report.append("PORTFOLIO RISK ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")

    report.append("KEY RISK METRICS")
    report.append("-" * 60)
    report.append(f"Portfolio Return (Annual):    {summary['portfolio_return_annual']:.2%}")
    report.append(f"Portfolio Volatility (Annual): {summary['portfolio_volatility_annual']:.2%}")
    report.append(f"Sharpe Ratio:                  {metrics.sharpe_ratio:.3f}")
    report.append(f"Sortino Ratio:                 {metrics.sortino_ratio:.3f}")
    report.append(f"Calmar Ratio:                  {metrics.calmar_ratio:.3f}")
    report.append(f"Maximum Drawdown:              {metrics.max_drawdown:.2%}")
    report.append("")

    report.append("VALUE AT RISK (VaR)")
    report.append("-" * 60)
    report.append(f"VaR 95%:  {metrics.var_95:.2%}  (Loss exceeded 5% of time)")
    report.append(f"VaR 99%:  {metrics.var_99:.2%}  (Loss exceeded 1% of time)")
    report.append(f"CVaR 95%: {metrics.cvar_95:.2%}  (Expected loss when VaR exceeded)")
    report.append(f"CVaR 99%: {metrics.cvar_99:.2%}")
    report.append("")

    report.append("TAIL RISK METRICS")
    report.append("-" * 60)
    report.append(f"Skewness: {metrics.skewness:.3f}  ({'Negative' if metrics.skewness < 0 else 'Positive'} skew)")
    report.append(f"Kurtosis: {metrics.kurtosis:.3f}  ({'Fat' if metrics.kurtosis > 0 else 'Thin'} tails)")
    report.append("")

    report.append("STRESS TEST RESULTS")
    report.append("-" * 60)
    for stress in summary['stress_tests']:
        report.append(f"{stress.scenario_name}:")
        report.append(f"  Portfolio Loss: {stress.portfolio_loss:.2f}%")
        report.append(f"  Worst Asset: {stress.worst_asset} ({stress.worst_asset_loss:.2f}%)")
        report.append("")

    report.append("COMPONENT VAR (Risk Contribution)")
    report.append("-" * 60)
    for asset, cvar in summary['component_var'].items():
        report.append(f"{asset}: {cvar:.4f}")
    report.append("")

    report.append("=" * 60)

    return "\n".join(report)
