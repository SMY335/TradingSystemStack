"""
Portfolio Manager
Orchestrates portfolio construction, optimization, and rebalancing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from .models import Portfolio, Asset, Position, AssetType
from .optimizer import PortfolioOptimizer, OptimizationMethod, RiskMeasure

logger = logging.getLogger(__name__)


@dataclass
class RebalancingConfig:
    """Configuration for portfolio rebalancing"""
    frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    threshold: float = 0.05  # Trigger rebalancing if weight deviation > 5%
    optimization_method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
    risk_measure: RiskMeasure = RiskMeasure.MV
    lookback_period: int = 90  # Days of historical data for optimization


class PortfolioManager:
    """
    Manages multi-asset portfolios with automatic rebalancing

    Features:
    - Multi-asset portfolio management
    - Automatic rebalancing based on time or drift
    - Portfolio optimization using Riskfolio-Lib
    - Performance tracking and attribution
    - Risk monitoring

    Example:
        manager = PortfolioManager(
            portfolio=my_portfolio,
            data_fetcher=fetch_data_function
        )

        # Optimize and rebalance
        trades = manager.rebalance()
        manager.execute_trades(trades)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        data_fetcher: Callable[[List[str], int], pd.DataFrame],
        optimizer: Optional[PortfolioOptimizer] = None,
        rebalancing_config: Optional[RebalancingConfig] = None
    ):
        """
        Initialize Portfolio Manager

        Args:
            portfolio: Portfolio to manage
            data_fetcher: Function that fetches historical data
                         Signature: (symbols: List[str], days: int) -> pd.DataFrame
            optimizer: Portfolio optimizer (creates default if None)
            rebalancing_config: Rebalancing configuration
        """
        # Validation
        if portfolio is None:
            raise ValueError("portfolio cannot be None")
        if not isinstance(portfolio, Portfolio):
            raise TypeError(f"portfolio must be Portfolio instance, got {type(portfolio).__name__}")

        if data_fetcher is None:
            raise ValueError("data_fetcher cannot be None")
        if not callable(data_fetcher):
            raise TypeError("data_fetcher must be callable")

        if optimizer is not None and not isinstance(optimizer, PortfolioOptimizer):
            raise TypeError(f"optimizer must be PortfolioOptimizer, got {type(optimizer).__name__}")

        if rebalancing_config is not None and not isinstance(rebalancing_config, RebalancingConfig):
            raise TypeError(f"rebalancing_config must be RebalancingConfig, got {type(rebalancing_config).__name__}")

        self.portfolio = portfolio
        self.data_fetcher = data_fetcher
        self.optimizer = optimizer or PortfolioOptimizer()
        self.rebalancing_config = rebalancing_config or RebalancingConfig()
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_history: List[Dict] = []

        logger.info(f"PortfolioManager initialized with {len(portfolio.positions)} positions")

    def should_rebalance(self) -> bool:
        """
        Check if portfolio should be rebalanced

        Rebalancing triggers:
        1. Time-based: frequency reached
        2. Drift-based: weights deviated beyond threshold
        """

        # Check time-based trigger
        if self.last_rebalance is None:
            return True

        days_since_rebalance = (datetime.now() - self.last_rebalance).days

        frequency_days = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }

        if days_since_rebalance >= frequency_days.get(self.rebalancing_config.frequency, 30):
            return True

        # Check drift-based trigger
        if self._check_drift_trigger():
            return True

        return False

    def _check_drift_trigger(self) -> bool:
        """Check if portfolio weights have drifted beyond threshold"""

        if not hasattr(self, '_target_weights') or self._target_weights is None:
            return False

        current_weights = self.portfolio.weights
        threshold = self.rebalancing_config.threshold

        for symbol, target_weight in self._target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            if abs(current_weight - target_weight) > threshold:
                return True

        return False

    def rebalance(self, force: bool = False) -> Dict[str, Dict]:
        """
        Rebalance portfolio to optimal weights

        Args:
            force: Force rebalancing even if not needed

        Returns:
            Dictionary with rebalancing trades and metadata
        """

        if not force and not self.should_rebalance():
            return {
                'rebalanced': False,
                'reason': 'No rebalancing needed',
                'trades': {}
            }

        # Get symbols to optimize
        symbols = [pos.asset.symbol for pos in self.portfolio.positions]

        if len(symbols) < 2:
            return {
                'rebalanced': False,
                'reason': 'Need at least 2 assets for optimization',
                'trades': {}
            }

        # Fetch historical data
        returns_df = self._fetch_returns_data(symbols)

        if returns_df.empty:
            return {
                'rebalanced': False,
                'reason': 'Failed to fetch historical data',
                'trades': {}
            }

        # Optimize portfolio
        optimal_weights = self.optimizer.optimize(
            returns_df,
            method=self.rebalancing_config.optimization_method,
            risk_measure=self.rebalancing_config.risk_measure
        )

        # Store target weights
        self._target_weights = optimal_weights

        # Calculate trades needed
        current_weights = self.portfolio.weights
        trades = self.optimizer.rebalancing_signals(
            current_weights,
            optimal_weights,
            threshold=0.01  # Trade even small deviations during rebalancing
        )

        # Calculate portfolio metrics before and after
        current_metrics = self.optimizer.calculate_portfolio_metrics(
            current_weights,
            returns_df
        )

        target_metrics = self.optimizer.calculate_portfolio_metrics(
            optimal_weights,
            returns_df
        )

        # Record rebalancing event
        rebalance_event = {
            'timestamp': datetime.now(),
            'current_weights': current_weights.copy(),
            'target_weights': optimal_weights.copy(),
            'trades': trades.copy(),
            'current_metrics': current_metrics,
            'target_metrics': target_metrics
        }

        self.rebalance_history.append(rebalance_event)
        self.last_rebalance = datetime.now()

        return {
            'rebalanced': True,
            'trades': trades,
            'current_weights': current_weights,
            'target_weights': optimal_weights,
            'current_metrics': current_metrics,
            'target_metrics': target_metrics,
            'improvement': {
                'sharpe_ratio': target_metrics['sharpe_ratio'] - current_metrics['sharpe_ratio'],
                'volatility': target_metrics['volatility'] - current_metrics['volatility']
            }
        }

    def _fetch_returns_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch and calculate returns data for optimization"""

        try:
            # Fetch price data
            prices_df = self.data_fetcher(
                symbols,
                self.rebalancing_config.lookback_period
            )

            if prices_df.empty:
                return pd.DataFrame()

            # Calculate returns
            returns_df = prices_df.pct_change().dropna()

            return returns_df

        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()

    def execute_trades(self, trades: Dict[str, float], current_prices: Dict[str, float]) -> Dict:
        """
        Execute rebalancing trades

        Args:
            trades: Dictionary of trades (symbol -> weight change)
            current_prices: Current prices for each symbol

        Returns:
            Execution summary
        """

        total_value = self.portfolio.total_value
        executed_trades = []

        for symbol, weight_change in trades.items():
            # Calculate dollar amount to trade
            dollar_amount = weight_change * total_value

            if symbol not in current_prices:
                logger.warning(f"No price for {symbol}, skipping trade")
                continue

            price = current_prices[symbol]

            # Calculate quantity
            quantity = dollar_amount / price

            # Get or create position
            position = self.portfolio.get_position(symbol)

            if position:
                # Update existing position
                old_quantity = position.quantity
                position.quantity += quantity

                if position.quantity <= 0:
                    # Close position
                    self.portfolio.remove_position(symbol)
                    self.portfolio.cash += old_quantity * price
                else:
                    # Adjust cash
                    self.portfolio.cash -= quantity * price

            else:
                # Create new position
                if quantity > 0:
                    asset = Asset(symbol=symbol, asset_type=AssetType.CRYPTO)
                    new_position = Position(
                        asset=asset,
                        quantity=quantity,
                        entry_price=price,
                        entry_date=datetime.now(),
                        current_price=price
                    )
                    self.portfolio.add_position(new_position)
                    self.portfolio.cash -= quantity * price

            executed_trades.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'dollar_amount': dollar_amount,
                'weight_change': weight_change
            })

        return {
            'executed': len(executed_trades),
            'trades': executed_trades,
            'new_cash': self.portfolio.cash,
            'new_total_value': self.portfolio.total_value
        }

    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""

        summary = self.portfolio.to_dict()

        # Add rebalancing info
        summary['rebalancing'] = {
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'should_rebalance': self.should_rebalance(),
            'rebalance_count': len(self.rebalance_history),
            'config': {
                'frequency': self.rebalancing_config.frequency,
                'threshold': self.rebalancing_config.threshold,
                'method': self.rebalancing_config.optimization_method.value
            }
        }

        # Add target weights if available
        if hasattr(self, '_target_weights') and self._target_weights:
            summary['target_weights'] = self._target_weights

        return summary

    def analyze_performance(self, lookback_days: int = 30) -> Dict:
        """
        Analyze portfolio performance over time

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Performance analysis
        """

        symbols = [pos.asset.symbol for pos in self.portfolio.positions]

        if not symbols:
            return {'error': 'No positions in portfolio'}

        # Fetch historical data
        returns_df = self._fetch_returns_data(symbols)

        if returns_df.empty:
            return {'error': 'Failed to fetch historical data'}

        # Calculate current portfolio metrics
        current_weights = self.portfolio.weights

        # Remove CASH from weights for returns calculation
        weights_without_cash = {k: v for k, v in current_weights.items() if k != 'CASH'}

        # Normalize weights
        total_weight = sum(weights_without_cash.values())
        if total_weight > 0:
            weights_without_cash = {k: v/total_weight for k, v in weights_without_cash.items()}

        metrics = self.optimizer.calculate_portfolio_metrics(
            weights_without_cash,
            returns_df
        )

        # Add portfolio specific info
        metrics['total_value'] = self.portfolio.total_value
        metrics['total_return_pct'] = self.portfolio.total_return_pct
        metrics['cash_ratio'] = current_weights.get('CASH', 0.0)

        return metrics

    def optimize_and_display_frontier(
        self,
        points: int = 20
    ) -> Dict:
        """
        Calculate and return efficient frontier data

        Args:
            points: Number of points on the frontier

        Returns:
            Efficient frontier data for visualization
        """

        symbols = [pos.asset.symbol for pos in self.portfolio.positions]

        if len(symbols) < 2:
            return {'error': 'Need at least 2 assets for efficient frontier'}

        # Fetch returns
        returns_df = self._fetch_returns_data(symbols)

        if returns_df.empty:
            return {'error': 'Failed to fetch historical data'}

        # Calculate efficient frontier
        risks, returns, weights_list = self.optimizer.efficient_frontier(
            returns_df,
            risk_measure=self.rebalancing_config.risk_measure,
            points=points
        )

        return {
            'risks': risks.tolist(),
            'returns': returns.tolist(),
            'weights': weights_list,
            'current_portfolio': self.optimizer.calculate_portfolio_metrics(
                self.portfolio.weights,
                returns_df
            )
        }
