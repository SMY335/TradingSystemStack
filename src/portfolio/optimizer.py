"""
Portfolio Optimizer using Riskfolio-Lib
Implements various optimization strategies:
- Mean-Variance (Markowitz)
- Risk Parity
- Maximum Sharpe Ratio
- Minimum Volatility
- Maximum Diversification
"""

import pandas as pd
import numpy as np
import riskfolio as rp
from typing import Dict, List, Optional, Tuple
from enum import Enum


class OptimizationMethod(Enum):
    """Available optimization methods"""
    MEAN_VARIANCE = "MV"  # Mean-Variance (Markowitz)
    MIN_VOLATILITY = "MinVol"  # Minimum Volatility
    MAX_SHARPE = "MaxSharpe"  # Maximum Sharpe Ratio
    RISK_PARITY = "RP"  # Risk Parity
    MAX_DIVERSIFICATION = "MaxDiv"  # Maximum Diversification
    EQUAL_WEIGHT = "EW"  # Equal Weight (naive)


class RiskMeasure(Enum):
    """Risk measures for optimization"""
    MV = "MV"  # Standard Deviation (Mean-Variance)
    MAD = "MAD"  # Mean Absolute Deviation
    MSV = "MSV"  # Semi Standard Deviation
    CVaR = "CVaR"  # Conditional Value at Risk
    EVaR = "EVaR"  # Entropic Value at Risk
    CDaR = "CDaR"  # Conditional Drawdown at Risk


class PortfolioOptimizer:
    """
    Portfolio Optimizer using Riskfolio-Lib

    Example:
        optimizer = PortfolioOptimizer()
        returns_df = pd.DataFrame(...)  # Historical returns
        weights = optimizer.optimize(
            returns_df,
            method=OptimizationMethod.MAX_SHARPE,
            risk_measure=RiskMeasure.MV
        )
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        frequency: int = 252,  # Trading days per year
        alpha: float = 0.05  # Significance level for VaR/CVaR
    ):
        """
        Initialize optimizer

        Args:
            risk_free_rate: Risk-free rate (annualized)
            frequency: Number of trading periods per year (252 for daily, 12 for monthly)
            alpha: Significance level for risk measures (e.g., 0.05 for 95% VaR)
        """
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.alpha = alpha

    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        risk_measure: RiskMeasure = RiskMeasure.MV,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights

        Args:
            returns: DataFrame with historical returns (columns = assets, rows = time periods)
            method: Optimization method to use
            risk_measure: Risk measure for optimization
            target_return: Target return for mean-variance optimization
            max_volatility: Maximum allowed volatility
            constraints: Additional constraints (min_weight, max_weight, etc.)

        Returns:
            Dictionary with asset symbols as keys and optimal weights as values
        """

        # Handle equal weight case (doesn't need optimization)
        if method == OptimizationMethod.EQUAL_WEIGHT:
            return self._equal_weight(returns.columns.tolist())

        # Create Portfolio object from Riskfolio-Lib
        port = rp.Portfolio(returns=returns)

        # Calculate expected returns and covariance
        port.assets_stats(method_mu='hist', method_cov='hist')

        # Apply constraints if provided
        if constraints:
            self._apply_constraints(port, constraints)

        # Optimize based on method
        if method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity_optimization(port)
        elif method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance_optimization(
                port, risk_measure, target_return, max_volatility
            )
        elif method == OptimizationMethod.MIN_VOLATILITY:
            weights = self._min_volatility_optimization(port, risk_measure)
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe_optimization(port, risk_measure)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification_optimization(port)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Convert to dictionary and clean small weights
        weights_dict = weights.to_dict()
        weights_dict = {k: v for k, v in weights_dict.items() if abs(v) > 1e-4}

        return weights_dict

    def _equal_weight(self, assets: List[str]) -> Dict[str, float]:
        """Equal weight portfolio (1/N)"""
        n = len(assets)
        return {asset: 1.0 / n for asset in assets}

    def _risk_parity_optimization(self, port: rp.Portfolio) -> pd.Series:
        """Risk Parity optimization - equal risk contribution"""
        try:
            port.rp = 'vol'  # Risk parity by volatility
            weights = port.rp_optimization(
                model='Classic',
                rm='MV',
                hist=True
            )
            if weights is None:
                print("Warning: Risk Parity optimization failed, falling back to equal weight")
                return pd.Series({col: 1.0/len(port.returns.columns) for col in port.returns.columns})
            return weights.iloc[:, 0]
        except Exception as e:
            print(f"Warning: Optimization error: {e}, falling back to equal weight")
            return pd.Series({col: 1.0/len(port.returns.columns) for col in port.returns.columns})

    def _mean_variance_optimization(
        self,
        port: rp.Portfolio,
        risk_measure: RiskMeasure,
        target_return: Optional[float],
        max_volatility: Optional[float]
    ) -> pd.Series:
        """Mean-Variance optimization (Markowitz)"""

        # Set objective function
        if target_return is not None:
            port.lowerret = target_return

        weights = port.optimization(
            model='Classic',
            rm=risk_measure.value,
            obj='Sharpe',
            hist=True,
            rf=self.risk_free_rate,
            l=0  # Risk aversion parameter
        )
        return weights.iloc[:, 0]

    def _min_volatility_optimization(
        self,
        port: rp.Portfolio,
        risk_measure: RiskMeasure
    ) -> pd.Series:
        """Minimum volatility portfolio"""
        try:
            weights = port.optimization(
                model='Classic',
                rm=risk_measure.value,
                obj='MinRisk',
                hist=True,
                rf=self.risk_free_rate,
                l=0
            )
            if weights is None:
                print("Warning: Min Volatility optimization failed, falling back to equal weight")
                return pd.Series({col: 1.0/len(port.returns.columns) for col in port.returns.columns})
            return weights.iloc[:, 0]
        except Exception as e:
            print(f"Warning: Optimization error: {e}, falling back to equal weight")
            return pd.Series({col: 1.0/len(port.returns.columns) for col in port.returns.columns})

    def _max_sharpe_optimization(
        self,
        port: rp.Portfolio,
        risk_measure: RiskMeasure
    ) -> pd.Series:
        """Maximum Sharpe Ratio portfolio"""
        try:
            weights = port.optimization(
                model='Classic',
                rm=risk_measure.value,
                obj='Sharpe',
                hist=True,
                rf=self.risk_free_rate,
                l=0
            )
            if weights is None:
                # Fallback to minimum volatility if Max Sharpe fails
                print("Warning: Max Sharpe optimization failed, falling back to Min Volatility")
                return self._min_volatility_optimization(port, risk_measure)
            return weights.iloc[:, 0]
        except Exception as e:
            print(f"Warning: Optimization error: {e}, falling back to equal weight")
            return pd.Series({col: 1.0/len(port.returns.columns) for col in port.returns.columns})

    def _max_diversification_optimization(self, port: rp.Portfolio) -> pd.Series:
        """Maximum Diversification portfolio"""
        weights = port.optimization(
            model='Classic',
            rm='MV',
            obj='Utility',
            hist=True,
            rf=self.risk_free_rate,
            l=0
        )
        return weights.iloc[:, 0]

    def _apply_constraints(self, port: rp.Portfolio, constraints: Dict) -> None:
        """Apply portfolio constraints"""

        if 'min_weight' in constraints:
            port.lowerbound = constraints['min_weight']

        if 'max_weight' in constraints:
            port.upperbound = constraints['max_weight']

        if 'budget' in constraints:
            port.budget = constraints['budget']

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        risk_measure: RiskMeasure = RiskMeasure.MV,
        points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        Calculate efficient frontier

        Args:
            returns: Historical returns DataFrame
            risk_measure: Risk measure to use
            points: Number of points on the frontier

        Returns:
            Tuple of (risks, returns, weights) for each point on frontier
        """

        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist')

        frontier = port.efficient_frontier(
            model='Classic',
            rm=risk_measure.value,
            points=points,
            rf=self.risk_free_rate,
            hist=True
        )

        risks = []
        rets = []
        weights_list = []

        for i in range(frontier.shape[1]):
            w = frontier.iloc[:, i]

            try:
                # Reindex weights to match portfolio assets
                w_reindexed = w.reindex(port.mu.index, fill_value=0.0)

                # Use pandas dot product for automatic index alignment
                portfolio_return = w_reindexed.dot(port.mu)
                portfolio_risk = np.sqrt(w_reindexed.dot(port.cov).dot(w_reindexed))

                risks.append(portfolio_risk)
                rets.append(portfolio_return)
                weights_list.append(w.to_dict())
            except Exception as e:
                print(f"Warning: Error calculating frontier point {i}: {e}")
                continue

        return np.array(risks), np.array(rets), weights_list

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics

        Args:
            weights: Portfolio weights
            returns: Historical returns

        Returns:
            Dictionary with performance metrics
        """

        # Convert weights to series
        weights_series = pd.Series(weights)

        # Align with returns columns
        weights_series = weights_series.reindex(returns.columns, fill_value=0)

        # Calculate portfolio returns
        portfolio_returns = (returns * weights_series).sum(axis=1)

        # Calculate metrics
        mean_return = portfolio_returns.mean() * self.frequency
        volatility = portfolio_returns.std() * np.sqrt(self.frequency)
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Calculate max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate VaR and CVaR
        var_95 = portfolio_returns.quantile(self.alpha)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        return {
            'expected_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': self._calculate_sortino(portfolio_returns),
            'calmar_ratio': mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside risk adjusted return)"""
        mean_return = returns.mean() * self.frequency
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = downside_returns.std() * np.sqrt(self.frequency)

        if downside_std == 0:
            return float('inf')

        return (mean_return - self.risk_free_rate) / downside_std

    def rebalancing_signals(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate rebalancing signals

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Minimum deviation to trigger rebalancing (e.g., 0.05 = 5%)

        Returns:
            Dictionary with rebalancing trades (positive = buy, negative = sell)
        """

        trades = {}

        # All assets from both portfolios
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            deviation = target - current

            # Only trade if deviation exceeds threshold
            if abs(deviation) > threshold:
                trades[asset] = deviation

        return trades
