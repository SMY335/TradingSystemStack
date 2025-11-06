"""
Factory for creating backtesting engines
"""
import logging
from typing import Union

from .base_engine import BaseBacktestEngine, BacktestConfig, BacktestEngine
from .backtrader_engine import BacktraderEngine
from .vectorbt_engine import VectorBTEngine

logger = logging.getLogger(__name__)


def create_backtest_engine(
    engine_type: Union[str, BacktestEngine] = BacktestEngine.BACKTRADER,
    config: BacktestConfig = None
) -> BaseBacktestEngine:
    """
    Factory function to create backtesting engines
    
    Args:
        engine_type: Type of engine ('backtrader' or 'vectorbt')
        config: Backtest configuration (uses defaults if None)
    
    Returns:
        BaseBacktestEngine instance
    
    Example:
        >>> config = BacktestConfig(initial_capital=50000, fees_pct=0.1)
        >>> engine = create_backtest_engine('backtrader', config)
        >>> result = engine.run(strategy, data)
    """
    if config is None:
        config = BacktestConfig()
    
    # Convert string to enum
    if isinstance(engine_type, str):
        engine_type = BacktestEngine(engine_type.lower())
    
    logger.info(f"Creating {engine_type.value} backtesting engine")
    
    if engine_type == BacktestEngine.BACKTRADER:
        return BacktraderEngine(config)
    
    elif engine_type == BacktestEngine.VECTORBT:
        return VectorBTEngine(config)
    
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


# Convenience functions
def backtrader_engine(config: BacktestConfig = None) -> BacktraderEngine:
    """Create Backtrader engine"""
    return create_backtest_engine(BacktestEngine.BACKTRADER, config)


def vectorbt_engine(config: BacktestConfig = None) -> VectorBTEngine:
    """Create VectorBT engine"""
    return create_backtest_engine(BacktestEngine.VECTORBT, config)
