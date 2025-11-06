"""
Backtrader Implementation of Unified Backtesting Engine
"""
import logging
import pandas as pd
import backtrader as bt
from datetime import datetime
from typing import Optional

from .base_engine import BaseBacktestEngine, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


class BacktraderEngine(BaseBacktestEngine):
    """
    Backtrader-based backtesting engine
    Provides detailed event-driven backtesting
    """
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        logger.info(f"Initialized BacktraderEngine with capital=${config.initial_capital}")
    
    def get_engine_name(self) -> str:
        return "Backtrader"
    
    def run(
        self,
        strategy,
        data: pd.DataFrame,
        **kwargs
    ) -> BacktestResult:
        """
        Run backtest using Backtrader
        
        Args:
            strategy: Strategy instance
            data: OHLCV DataFrame
            **kwargs: Additional Backtrader-specific options
        
        Returns:
            BacktestResult with standardized metrics
        """
        # Validate data
        self.validate_data(data)
        
        logger.info(f"Running Backtrader backtest: {strategy.name} on {len(data)} bars")
        
        # Create Cerebro instance
        cerebro = bt.Cerebro()
        
        # Set cash
        cerebro.broker.set_cash(self.config.initial_capital)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.config.fees_pct / 100)
        
        # Convert strategy to Backtrader format
        bt_strategy = self._convert_strategy(strategy)
        cerebro.addstrategy(bt_strategy)
        
        # Add data feed
        data_feed = self._create_datafeed(data)
        cerebro.adddata(data_feed)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        try:
            results = cerebro.run()
            strat = results[0]
            
            # Extract metrics
            result = self._extract_metrics(strat, data)
            
            logger.info(
                f"Backtest complete: Return={result.total_return_pct:.2f}%, "
                f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.total_trades}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            raise RuntimeError(f"Backtrader backtest failed: {e}") from e
    
    def _convert_strategy(self, strategy):
        """Convert our strategy format to Backtrader strategy"""
        
        class BTStrategy(bt.Strategy):
            def __init__(self):
                self.order = None
                self.dataclose = self.datas[0].close
                self.strategy_logic = strategy
            
            def next(self):
                # Get recent data for signal generation
                df = self._get_dataframe()
                
                # Generate signals
                entries, exits = self.strategy_logic.generate_signals(df)
                
                # Execute signals
                if entries.iloc[-1] and not self.position:
                    self.buy()
                elif exits.iloc[-1] and self.position:
                    self.sell()
            
            def _get_dataframe(self):
                """Convert Backtrader data to DataFrame"""
                data = []
                for i in range(-min(len(self), 100), 0):
                    data.append({
                        'open': self.datas[0].open[i],
                        'high': self.datas[0].high[i],
                        'low': self.datas[0].low[i],
                        'close': self.datas[0].close[i],
                        'volume': self.datas[0].volume[i]
                    })
                return pd.DataFrame(data)
        
        return BTStrategy
    
    def _create_datafeed(self, df: pd.DataFrame):
        """Create Backtrader data feed from DataFrame"""
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        
        # Create PandasData feed
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        return data_feed
    
    def _extract_metrics(self, strat, data: pd.DataFrame) -> BacktestResult:
        """Extract metrics from Backtrader strategy"""
        
        # Get analyzers
        trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        # Extract trade statistics
        total = trade_analyzer.get('total', {})
        won = trade_analyzer.get('won', {})
        lost = trade_analyzer.get('lost', {})
        
        total_trades = total.get('total', 0)
        winning_trades = won.get('total', 0)
        losing_trades = lost.get('total', 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = won.get('pnl', {}).get('average', 0)
        avg_loss = abs(lost.get('pnl', {}).get('average', 0))
        
        # Profit factor
        gross_profit = won.get('pnl', {}).get('total', 0)
        gross_loss = abs(lost.get('pnl', {}).get('total', 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Returns
        total_return = returns.get('rtot', 0) * 100
        annualized_return = returns.get('rnorm100', 0)
        
        # Risk metrics
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        sharpe_ratio = sharpe.get('sharperatio', 0) if sharpe.get('sharperatio') is not None else 0
        
        # Calculate additional metrics
        daily_returns = pd.Series([r for r in returns.get('daily', {}).values()])
        volatility = daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 0 else 0
        
        # Sortino
        downside = daily_returns[daily_returns < 0]
        downside_std = downside.std() * (252 ** 0.5) if len(downside) > 0 else volatility
        sortino = (annualized_return - 2) / downside_std if downside_std > 0 else 0
        
        # Calmar
        calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0
        
        # Build result
        result = BacktestResult(
            total_return_pct=total_return,
            annualized_return_pct=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd,
            volatility_pct=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            avg_win_pct=(avg_win / self.config.initial_capital * 100),
            avg_loss_pct=(avg_loss / self.config.initial_capital * 100),
            profit_factor=profit_factor,
            avg_trade_duration_hours=0,  # Not available from Backtrader
            exposure_time_pct=0,  # Calculate separately if needed
            sqn=None
        )
        
        return result
