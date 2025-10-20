"""
EMA Crossover Strategy
Buy when fast EMA crosses above slow EMA, sell when crosses below
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class EMAStrategy(BaseStrategy):
    """Exponential Moving Average Crossover Strategy"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        params = {'fast_period': fast_period, 'slow_period': slow_period}
        super().__init__("EMA Crossover", params)

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate EMA crossover signals"""
        close = df['close'] if 'close' in df.columns else df['Close']

        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']

        # Calculate EMAs
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        # Generate signals
        # Buy when fast crosses above slow
        entries = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))

        # Sell when fast crosses below slow
        exits = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        return entries, exits

    def get_description(self) -> str:
        return f"EMA({self.params['fast_period']}) crosses EMA({self.params['slow_period']})"

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            'fast_period': {
                'type': 'int',
                'min': 5,
                'max': 50,
                'default': 12,
                'label': 'Fast EMA Period'
            },
            'slow_period': {
                'type': 'int',
                'min': 20,
                'max': 200,
                'default': 26,
                'label': 'Slow EMA Period'
            }
        }
