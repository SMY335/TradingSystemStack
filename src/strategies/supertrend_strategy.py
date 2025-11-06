"""
SuperTrend Strategy
Trend-following indicator based on ATR (Average True Range)
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend Strategy - ATR-based trend following

    SuperTrend is a volatility-based indicator that provides clear buy/sell signals.
    It uses ATR (Average True Range) to calculate dynamic support/resistance levels.

    Formula:
    - Basic Upper Band = (High + Low) / 2 + (Multiplier × ATR)
    - Basic Lower Band = (High + Low) / 2 - (Multiplier × ATR)
    - Final Bands adjust based on previous close to create trailing stops

    Trading Logic:
    - BUY: Price closes above SuperTrend (trend changes to UP)
    - SELL: Price closes below SuperTrend (trend changes to DOWN)
    - The indicator acts as a trailing stop loss in trending markets

    Parameters:
    - period: ATR calculation period (default: 10)
    - multiplier: ATR multiplier for bands (default: 3.0)
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize SuperTrend Strategy

        Args:
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)
        """
        # Validation
        if not isinstance(period, int):
            raise TypeError(f"period must be int, got {type(period).__name__}")
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        if period > 50:
            raise ValueError(f"period cannot exceed 50, got {period}")

        if not isinstance(multiplier, (int, float)):
            raise TypeError(f"multiplier must be numeric, got {type(multiplier).__name__}")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        if multiplier > 10:
            raise ValueError(f"multiplier cannot exceed 10, got {multiplier}")

        params = {'period': period, 'multiplier': multiplier}
        super().__init__("SuperTrend", params)
        logger.debug(f"SuperTrendStrategy initialized: period={period}, multiplier={multiplier}")

    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            Series with ATR values
        """
        # Get price columns
        high = df['high'] if 'high' in df.columns else df.get('High', df['close'])
        low = df['low'] if 'low' in df.columns else df.get('Low', df['close'])
        close = df['close'] if 'close' in df.columns else df['Close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR as rolling mean of True Range
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (supertrend values, trend direction)
            trend: 1 = UP, -1 = DOWN
        """
        period = self.params['period']
        multiplier = self.params['multiplier']

        # Get price data
        high = df['high'] if 'high' in df.columns else df.get('High', df['close'])
        low = df['low'] if 'low' in df.columns else df.get('Low', df['close'])
        close = df['close'] if 'close' in df.columns else df['Close']

        # Calculate ATR
        atr = self.calculate_atr(df, period)

        # Calculate basic bands
        hl_avg = (high + low) / 2
        basic_upper = hl_avg + (multiplier * atr)
        basic_lower = hl_avg - (multiplier * atr)

        # Initialize final bands
        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()

        # Calculate final bands with trailing logic
        for i in range(period, len(df)):
            # Upper band: use basic upper unless close is above previous final upper
            if i > 0 and close.iloc[i-1] <= final_upper.iloc[i-1]:
                final_upper.iloc[i] = min(basic_upper.iloc[i], final_upper.iloc[i-1])
            else:
                final_upper.iloc[i] = basic_upper.iloc[i]

            # Lower band: use basic lower unless close is below previous final lower
            if i > 0 and close.iloc[i-1] >= final_lower.iloc[i-1]:
                final_lower.iloc[i] = max(basic_lower.iloc[i], final_lower.iloc[i-1])
            else:
                final_lower.iloc[i] = basic_lower.iloc[i]

        # Determine SuperTrend value and direction
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)

        # Initialize first value
        if close.iloc[period] > final_upper.iloc[period]:
            supertrend.iloc[period] = final_lower.iloc[period]
            trend.iloc[period] = 1  # UP
        else:
            supertrend.iloc[period] = final_upper.iloc[period]
            trend.iloc[period] = -1  # DOWN

        # Calculate for remaining periods
        for i in range(period + 1, len(df)):
            if close.iloc[i] > final_upper.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                trend.iloc[i] = 1  # UP
            elif close.iloc[i] < final_lower.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                trend.iloc[i] = -1  # DOWN
            else:
                # Continue previous trend
                supertrend.iloc[i] = supertrend.iloc[i-1]
                trend.iloc[i] = trend.iloc[i-1]

                # Update SuperTrend value based on trend
                if trend.iloc[i] == 1:
                    supertrend.iloc[i] = final_lower.iloc[i]
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]

        return supertrend, trend

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate SuperTrend signals

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        supertrend, trend = self.calculate_supertrend(df)

        # Entry signal: Trend changes from DOWN to UP
        entries = (trend == 1) & (trend.shift(1) == -1)

        # Exit signal: Trend changes from UP to DOWN
        exits = (trend == -1) & (trend.shift(1) == 1)

        return entries, exits

    def get_description(self) -> str:
        """Get strategy description"""
        return f"SuperTrend({self.params['period']}, {self.params['multiplier']}×ATR)"

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for optimization"""
        return {
            'period': {
                'type': 'int',
                'min': 7,
                'max': 20,
                'default': 10,
                'label': 'ATR Period'
            },
            'multiplier': {
                'type': 'float',
                'min': 1.0,
                'max': 5.0,
                'default': 3.0,
                'step': 0.5,
                'label': 'ATR Multiplier'
            }
        }

    def get_supertrend_for_plot(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get SuperTrend data for plotting

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dictionary with 'supertrend', 'trend', 'upper', 'lower'
        """
        supertrend, trend = self.calculate_supertrend(df)

        # Also calculate bands for visualization
        atr = self.calculate_atr(df, self.params['period'])
        close = df['close'] if 'close' in df.columns else df['Close']
        high = df['high'] if 'high' in df.columns else df.get('High', close)
        low = df['low'] if 'low' in df.columns else df.get('Low', close)

        hl_avg = (high + low) / 2
        upper = hl_avg + (self.params['multiplier'] * atr)
        lower = hl_avg - (self.params['multiplier'] * atr)

        return {
            'supertrend': supertrend,
            'trend': trend,
            'upper_band': upper,
            'lower_band': lower,
            'atr': atr
        }
