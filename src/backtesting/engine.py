"""
Backtesting Engine using VectorBT
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest trading strategies using VectorBT"""

    def __init__(
        self,
        initial_cash: float = 10000,
        fees: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtest engine

        Args:
            initial_cash: Starting capital
            fees: Trading fees as decimal (0.001 = 0.1%)
            slippage: Slippage as decimal (0.0005 = 0.05%)
        """
        # Validation
        if not isinstance(initial_cash, (int, float)):
            raise TypeError(f"initial_cash must be numeric, got {type(initial_cash).__name__}")
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        if not isinstance(fees, (int, float)):
            raise TypeError(f"fees must be numeric, got {type(fees).__name__}")
        if fees < 0 or fees > 1:
            raise ValueError(f"fees must be between 0 and 1 (0-100%), got {fees}")

        if not isinstance(slippage, (int, float)):
            raise TypeError(f"slippage must be numeric, got {type(slippage).__name__}")
        if slippage < 0 or slippage > 1:
            raise ValueError(f"slippage must be between 0 and 1 (0-100%), got {slippage}")

        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage

        logger.info(f"BacktestEngine initialized: cash=${initial_cash:,.2f}, fees={fees:.4f}, slippage={slippage:.4f}")

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame
    ) -> Tuple[vbt.Portfolio, Dict[str, Any]]:
        """
        Run backtest for a strategy

        Args:
            strategy: Strategy instance to backtest
            df: Price data DataFrame

        Returns:
            Tuple of (portfolio, kpis_dict)
        """
        # Generate signals
        entries, exits = strategy.generate_signals(df)

        # Get close prices
        close = df['close'] if 'close' in df.columns else df['Close']

        # Run backtest
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq='1h'  # Adjust based on your data
        )

        # Calculate KPIs
        kpis = self._calculate_kpis(portfolio)

        return portfolio, kpis

    def _calculate_kpis(self, portfolio: vbt.Portfolio) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        trades = portfolio.trades

        # Safely extract metrics
        try:
            total_return = portfolio.total_return()
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to calculate total_return: {e}")
            total_return = 0

        try:
            win_rate = trades.win_rate()
            if isinstance(win_rate, pd.Series):
                win_rate = win_rate.iloc[0] if len(win_rate) > 0 else 0
            win_rate = 0 if np.isnan(win_rate) else win_rate
        except (AttributeError, ValueError, KeyError, TypeError, IndexError) as e:
            logger.warning(f"Failed to calculate win_rate: {e}")
            win_rate = 0

        try:
            profit_factor = trades.profit_factor()
            if isinstance(profit_factor, pd.Series):
                profit_factor = profit_factor.iloc[0] if len(profit_factor) > 0 else 0
            profit_factor = 0 if np.isnan(profit_factor) or np.isinf(profit_factor) else profit_factor
        except (AttributeError, ValueError, KeyError, TypeError, IndexError) as e:
            logger.warning(f"Failed to calculate profit_factor: {e}")
            profit_factor = 0

        try:
            max_dd = portfolio.max_drawdown()
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to calculate max_drawdown: {e}")
            max_dd = 0

        try:
            sharpe = portfolio.sharpe_ratio()
        except (AttributeError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
            logger.warning(f"Failed to calculate sharpe_ratio: {e}")
            sharpe = 0

        try:
            num_trades = trades.count()
            if isinstance(num_trades, pd.Series):
                num_trades = int(num_trades.iloc[0]) if len(num_trades) > 0 else 0
        except (AttributeError, ValueError, KeyError, TypeError, IndexError) as e:
            logger.warning(f"Failed to calculate num_trades: {e}")
            num_trades = 0

        final_val = portfolio.final_value()

        kpis = {
            'total_return_pct': round(total_return * 100, 2),
            'total_trades': num_trades,
            'win_rate_pct': round(win_rate * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'final_value': round(final_val, 2),
            'total_pnl': round(final_val - self.initial_cash, 2),
        }

        return kpis

    def compare_strategies(
        self,
        strategies: list[BaseStrategy],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare multiple strategies

        Args:
            strategies: List of strategy instances
            df: Price data DataFrame

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for strategy in strategies:
            portfolio, kpis = self.run(strategy, df)
            kpis['strategy'] = strategy.name
            kpis['description'] = strategy.get_description()
            results.append(kpis)

        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)

        return comparison_df
