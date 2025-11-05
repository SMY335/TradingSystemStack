"""
RSI Strategy
Buy when RSI crosses below oversold threshold, sell when crosses above overbought
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    """Relative Strength Index Strategy"""

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        params = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought
        }
        super().__init__("RSI", params)

    def calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate RSI signals"""
        close = df['close'] if 'close' in df.columns else df['Close']

        period = self.params['period']
        oversold = self.params['oversold']
        overbought = self.params['overbought']

        # Calculate RSI
        rsi = self.calculate_rsi(close, period)

        # Generate signals
        # Buy when RSI crosses above oversold level (coming from below)
        entries = (rsi > oversold) & (rsi.shift(1) <= oversold)

        # Sell when RSI crosses below overbought level (coming from above)
        exits = (rsi < overbought) & (rsi.shift(1) >= overbought)

        return entries, exits

    def get_description(self) -> str:
        return f"RSI({self.params['period']}) with levels {self.params['oversold']}/{self.params['overbought']}"

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {
            'period': {
                'type': 'int',
                'min': 5,
                'max': 30,
                'default': 14,
                'label': 'RSI Period'
            },
            'oversold': {
                'type': 'int',
                'min': 10,
                'max': 40,
                'default': 30,
                'label': 'Oversold Level'
            },
            'overbought': {
                'type': 'int',
                'min': 60,
                'max': 90,
                'default': 70,
                'label': 'Overbought Level'
            }
        }
