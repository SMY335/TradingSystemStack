"""
MACD Strategy
Buy when MACD line crosses above signal line, sell when crosses below
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence Strategy"""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        params = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
        super().__init__("MACD", params)

    def calculate_macd(self, close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, signal line, and histogram"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate MACD crossover signals"""
        close = df['close'] if 'close' in df.columns else df['Close']

        fast = self.params['fast_period']
        slow = self.params['slow_period']
        signal = self.params['signal_period']

        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(close, fast, slow, signal)

        # Generate signals
        # Buy when MACD crosses above signal line
        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

        # Sell when MACD crosses below signal line
        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        return entries, exits

    def get_description(self) -> str:
        return f"MACD({self.params['fast_period']},{self.params['slow_period']},{self.params['signal_period']})"

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            'fast_period': {
                'type': 'int',
                'min': 5,
                'max': 20,
                'default': 12,
                'label': 'Fast Period'
            },
            'slow_period': {
                'type': 'int',
                'min': 20,
                'max': 50,
                'default': 26,
                'label': 'Slow Period'
            },
            'signal_period': {
                'type': 'int',
                'min': 5,
                'max': 15,
                'default': 9,
                'label': 'Signal Period'
            }
        }
