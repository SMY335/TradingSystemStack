"""
Base Backtesting Engine Interface
Provides unified interface for multiple backtesting frameworks
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class BacktestEngine(str, Enum):
    """Available backtesting engines"""
    VECTORBT = "vectorbt"
    BACKTRADER = "backtrader"


@dataclass
class BacktestResult:
    """
    Standardized backtest result format
    All engines must return this format
    """
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    volatility_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    
    # Trade metrics
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_trade_duration_hours: float
    
    # Advanced metrics
    exposure_time_pct: float
    sqn: Optional[float] = None  # System Quality Number
    
    # Raw data
    equity_curve: Optional[pd.Series] = None
    trades_df: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'volatility_pct': self.volatility_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': self.win_rate_pct,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'profit_factor': self.profit_factor,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'exposure_time_pct': self.exposure_time_pct,
            'sqn': self.sqn
        }


@dataclass
class BacktestConfig:
    """
    Standardized backtest configuration
    """
    initial_capital: float = 10000.0
    fees_pct: float = 0.1
    slippage_pct: float = 0.05
    
    # Validation
    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if self.fees_pct < 0 or self.fees_pct > 10:
            raise ValueError(f"fees_pct must be between 0 and 10, got {self.fees_pct}")
        
        if self.slippage_pct < 0 or self.slippage_pct > 5:
            raise ValueError(f"slippage_pct must be between 0 and 5, got {self.slippage_pct}")


class BaseBacktestEngine(ABC):
    """
    Abstract base class for backtesting engines
    All implementations must follow this interface
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine
        
        Args:
            config: Backtest configuration
        """
        self.config = config
    
    @abstractmethod
    def run(
        self,
        strategy,
        data: pd.DataFrame,
        **kwargs
    ) -> BacktestResult:
        """
        Run backtest for a strategy
        
        Args:
            strategy: Strategy instance to backtest
            data: Price data DataFrame with OHLCV columns
            **kwargs: Engine-specific parameters
        
        Returns:
            BacktestResult with standardized metrics
        """
        pass
    
    @abstractmethod
    def get_engine_name(self) -> str:
        """Return engine name"""
        pass
    
    def validate_data(self, df: pd.DataFrame):
        """
        Validate input data
        
        Args:
            df: DataFrame to validate
        
        Raises:
            ValueError: If data is invalid
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data: {len(df)} bars. Need at least 50.")
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calculate standardized metrics from equity curve and trades
        
        Args:
            equity_curve: Equity over time
            trades: DataFrame with trade details
            initial_capital: Starting capital
        
        Returns:
            Dictionary of metrics
        """
        import numpy as np
        
        # Returns
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        # Daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = excess_return / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()
        
        # Calmar ratio
        calmar = abs(annualized_return / max_dd) if max_dd != 0 else 0
        
        # Trade statistics
        if trades is not None and not trades.empty:
            total_trades = len(trades)
            winning = trades[trades['pnl'] > 0]
            losing = trades[trades['pnl'] < 0]
            
            winning_trades = len(winning)
            losing_trades = len(losing)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
            avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0
            
            profit_factor = (winning['pnl'].sum() / abs(losing['pnl'].sum())) if len(losing) > 0 else 0
            
            # Trade duration
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
                avg_duration = durations.mean()
            else:
                avg_duration = 0
        else:
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown_pct': max_dd * 100,
            'volatility_pct': volatility * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': (avg_win / initial_capital * 100) if initial_capital > 0 else 0,
            'avg_loss_pct': (avg_loss / initial_capital * 100) if initial_capital > 0 else 0,
            'profit_factor': profit_factor,
            'avg_trade_duration_hours': avg_duration
        }
