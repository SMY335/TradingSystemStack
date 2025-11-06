"""
Bollinger Bands Strategy
Buy when price touches lower band (oversold), sell when touches upper band (overbought)
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy - Volatility-based mean reversion

    Bollinger Bands consist of:
    - Middle Band: Simple Moving Average (SMA)
    - Upper Band: SMA + (std_dev × standard deviation)
    - Lower Band: SMA - (std_dev × standard deviation)

    Trading Logic:
    - BUY: Price touches or crosses below lower band (oversold)
    - SELL: Price touches or crosses above upper band (overbought)
    - Additional: Detect "squeeze" (low volatility) for potential breakouts

    Parameters:
    - period: Lookback period for SMA and standard deviation (default: 20)
    - std_dev: Number of standard deviations for bands (default: 2.0)
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands Strategy

        Args:
            period: Lookback period for bands (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        # Validation
        if not isinstance(period, int):
            raise TypeError(f"period must be int, got {type(period).__name__}")
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if period > 200:
            raise ValueError(f"period cannot exceed 200, got {period}")

        if not isinstance(std_dev, (int, float)):
            raise TypeError(f"std_dev must be numeric, got {type(std_dev).__name__}")
        if std_dev <= 0:
            raise ValueError(f"std_dev must be positive, got {std_dev}")
        if std_dev > 10:
            raise ValueError(f"std_dev cannot exceed 10, got {std_dev}")

        params = {'period': period, 'std_dev': std_dev}
        super().__init__("Bollinger Bands", params)
        logger.debug(f"BollingerBandsStrategy initialized: period={period}, std_dev={std_dev}")

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate Bollinger Bands signals

        Args:
            df: DataFrame with 'close' prices

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        close = df['close'] if 'close' in df.columns else df['Close']

        period = self.params['period']
        std_dev = self.params['std_dev']

        # Calculate Bollinger Bands
        sma = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper_band = sma + (std_dev * rolling_std)
        lower_band = sma - (std_dev * rolling_std)

        # Calculate band width (for squeeze detection)
        band_width = (upper_band - lower_band) / sma

        # Entry signal: Price touches or crosses below lower band
        # Additional condition: Not already in a position (avoid multiple entries)
        price_below_lower = close <= lower_band
        was_above_lower = close.shift(1) > lower_band.shift(1)
        entries = price_below_lower & was_above_lower

        # Exit signal: Price touches or crosses above upper band
        price_above_upper = close >= upper_band
        was_below_upper = close.shift(1) < upper_band.shift(1)
        exits = price_above_upper & was_below_upper

        # Alternative exit: Price crosses back above middle band (conservative)
        # Can be enabled for more conservative exits
        # exits = exits | (close > sma) & (close.shift(1) <= sma.shift(1))

        return entries, exits

    def get_description(self) -> str:
        """Get strategy description"""
        return f"Bollinger Bands({self.params['period']}, {self.params['std_dev']}σ)"

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for optimization"""
        return {
            'period': {
                'type': 'int',
                'min': 10,
                'max': 50,
                'default': 20,
                'label': 'Period (SMA window)'
            },
            'std_dev': {
                'type': 'float',
                'min': 1.0,
                'max': 4.0,
                'default': 2.0,
                'step': 0.5,
                'label': 'Standard Deviation Multiplier'
            }
        }

    def calculate_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands for visualization

        Args:
            df: DataFrame with 'close' prices

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'width' bands
        """
        close = df['close'] if 'close' in df.columns else df['Close']

        period = self.params['period']
        std_dev = self.params['std_dev']

        sma = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper_band = sma + (std_dev * rolling_std)
        lower_band = sma - (std_dev * rolling_std)
        band_width = (upper_band - lower_band) / sma

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': band_width
        }

    def detect_squeeze(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Detect Bollinger Band "squeeze" - periods of low volatility

        A squeeze occurs when bands are narrow, indicating low volatility.
        This often precedes a significant price move (breakout).

        Args:
            df: DataFrame with 'close' prices
            threshold: Band width threshold for squeeze (default: 0.1 = 10%)

        Returns:
            Series of boolean values indicating squeeze periods
        """
        bands = self.calculate_bands(df)
        band_width = bands['width']

        # Squeeze = band width below threshold
        # Also compare to recent average to detect relative narrowing
        avg_width = band_width.rolling(window=50).mean()

        squeeze = (band_width < threshold) | (band_width < avg_width * 0.5)

        return squeeze.fillna(False)
