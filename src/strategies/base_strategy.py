"""
Base Strategy Class - All trading strategies inherit from this
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""

    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate buy and sell signals based on the strategy logic

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            Tuple of (entries, exits) where:
                - entries: Boolean series where True indicates buy signal
                - exits: Boolean series where True indicates sell signal
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a human-readable description of the strategy"""
        pass

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Define parameter schema for the strategy

        Returns:
            Dict with parameter names as keys and their config as values
            Example: {
                'fast_period': {'type': 'int', 'min': 1, 'max': 100, 'default': 12},
                'slow_period': {'type': 'int', 'min': 1, 'max': 200, 'default': 26}
            }
        """
        return {}

    def __str__(self):
        return f"{self.name} - {self.get_description()}"
