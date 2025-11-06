"""
Trading Strategies Module
"""
from .base_strategy import BaseStrategy
from .ema_strategy import EMAStrategy
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerBandsStrategy
from .supertrend_strategy import SuperTrendStrategy
from .ichimoku_strategy import IchimokuStrategy

# Registry of all available strategies
AVAILABLE_STRATEGIES = {
    'EMA Crossover': EMAStrategy,
    'RSI': RSIStrategy,
    'MACD': MACDStrategy,
    'Bollinger Bands': BollingerBandsStrategy,
    'SuperTrend': SuperTrendStrategy,
    'Ichimoku Cloud': IchimokuStrategy,
}

__all__ = [
    'BaseStrategy',
    'EMAStrategy',
    'RSIStrategy',
    'MACDStrategy',
    'BollingerBandsStrategy',
    'SuperTrendStrategy',
    'IchimokuStrategy',
    'AVAILABLE_STRATEGIES'
]
