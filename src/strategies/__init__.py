"""
Trading Strategies Module
"""
from .base_strategy import BaseStrategy
from .ema_strategy import EMAStrategy
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy

# Registry of all available strategies
AVAILABLE_STRATEGIES = {
    'EMA Crossover': EMAStrategy,
    'RSI': RSIStrategy,
    'MACD': MACDStrategy,
}

__all__ = [
    'BaseStrategy',
    'EMAStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'AVAILABLE_STRATEGIES'
]
