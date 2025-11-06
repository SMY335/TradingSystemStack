"""
Unified Backtesting Module
Supports multiple backtesting engines with consistent interface
"""

# New unified interface
from .base_engine import (
    BaseBacktestEngine,
    BacktestResult,
    BacktestConfig,
    BacktestEngine
)
from .engine_factory import (
    create_backtest_engine,
    backtrader_engine,
    vectorbt_engine
)
from .backtrader_engine import BacktraderEngine
from .vectorbt_engine import VectorBTEngine

# Legacy imports (kept for backward compatibility)
from .advanced_engine import (
    AdvancedBacktestEngine,
    BacktestConfig as AdvancedBacktestConfig,
    TransactionCosts
)
from .metrics import AdvancedMetrics
from .monte_carlo import MonteCarloSimulator

__all__ = [
    # New unified interface
    'BaseBacktestEngine',
    'BacktestResult',
    'BacktestConfig',
    'BacktestEngine',
    'create_backtest_engine',
    'backtrader_engine',
    'vectorbt_engine',
    'BacktraderEngine',
    'VectorBTEngine',
    # Legacy (backward compatibility)
    'AdvancedBacktestEngine',
    'AdvancedBacktestConfig',
    'TransactionCosts',
    'AdvancedMetrics',
    'MonteCarloSimulator'
]
