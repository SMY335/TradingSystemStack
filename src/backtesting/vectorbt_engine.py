"""
VectorBT Implementation of Unified Backtesting Engine
Fast vectorized backtesting for quick iterations
"""
import logging
import pandas as pd
import numpy as np

from .base_engine import BaseBacktestEngine, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


class VectorBTEngine(BaseBacktestEngine):
    """
    VectorBT-based backtesting engine
    Provides fast vectorized backtesting
    """
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        logger.info(f"Initialized VectorBTEngine with capital=${config.initial_capital}")
    
    def get_engine_name(self) -> str:
        return "VectorBT"
    
    def run(
        self,
        strategy,
        data: pd.DataFrame,
        **kwargs
    ) -> BacktestResult:
        """
        Run backtest using VectorBT
        
        Args:
            strategy: Strategy instance
            data: OHLCV DataFrame
            **kwargs: Additional VectorBT-specific options
        
        Returns:
            BacktestResult with standardized metrics
        """
        # Validate data
        self.validate_data(data)
        
        logger.info(f"Running VectorBT backtest: {strategy.name} on {len(data)} bars")
        
        try:
            # Try to import vectorbt
            try:
                import vectorbt as vbt
            except ImportError:
                logger.error("VectorBT not installed. Install with: pip install vectorbt")
                raise ImportError(
                    "VectorBT is required for this engine. Install with: pip install vectorbt"
                )
            
            # Generate signals
            entries, exits = strategy.generate_signals(data)
            
            # Run VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                init_cash=self.config.initial_capital,
                fees=self.config.fees_pct / 100,
                slippage=self.config.slippage_pct / 100
            )
            
            # Extract metrics
            result = self._extract_metrics(portfolio, data)
            
            logger.info(
                f"Backtest complete: Return={result.total_return_pct:.2f}%, "
                f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.total_trades}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"VectorBT backtest failed: {e}", exc_info=True)
            raise RuntimeError(f"VectorBT backtest failed: {e}") from e
    
    def _extract_metrics(self, portfolio, data: pd.DataFrame) -> BacktestResult:
        """Extract standardized metrics from VectorBT portfolio"""
        
        # Get trades
        trades = portfolio.trades.records_readable
        
        # Calculate metrics using base class method
        equity_curve = portfolio.value()
        
        if not trades.empty:
            trades_df = pd.DataFrame({
                'entry_time': pd.to_datetime(trades['Entry Timestamp']),
                'exit_time': pd.to_datetime(trades['Exit Timestamp']),
                'pnl': trades['PnL']
            })
        else:
            trades_df = pd.DataFrame()
        
        metrics = self.calculate_metrics(
            equity_curve=equity_curve,
            trades=trades_df,
            initial_capital=self.config.initial_capital
        )
        
        # Get exposure time
        try:
            exposure_time = (portfolio.positions.duration.sum() / len(data) * 100)
        except:
            exposure_time = 0
        
        # Build result
        result = BacktestResult(
            total_return_pct=metrics['total_return_pct'],
            annualized_return_pct=metrics['annualized_return_pct'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            calmar_ratio=metrics['calmar_ratio'],
            max_drawdown_pct=metrics['max_drawdown_pct'],
            volatility_pct=metrics['volatility_pct'],
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate_pct=metrics['win_rate_pct'],
            avg_win_pct=metrics['avg_win_pct'],
            avg_loss_pct=metrics['avg_loss_pct'],
            profit_factor=metrics['profit_factor'],
            avg_trade_duration_hours=metrics['avg_trade_duration_hours'],
            exposure_time_pct=exposure_time,
            equity_curve=equity_curve,
            trades_df=trades_df
        )
        
        return result
