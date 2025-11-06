"""
Ichimoku Cloud Strategy
Complete Japanese trading system with 5 lines providing support/resistance and trend
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class IchimokuStrategy(BaseStrategy):
    """
    Ichimoku Kinko Hyo (Equilibrium Chart) Strategy

    A comprehensive Japanese trading system developed by Goichi Hosoda.
    Provides support/resistance, trend direction, and momentum in one view.

    Components:
    1. Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
       - Fast moving line showing short-term momentum

    2. Kijun-sen (Base Line): (26-period high + 26-period low) / 2
       - Slower line showing medium-term trend

    3. Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted +26
       - Forms one edge of the cloud

    4. Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted +26
       - Forms other edge of the cloud

    5. Chikou Span (Lagging Span): Close price shifted -26
       - Momentum and confirmation indicator

    The "Cloud" (Kumo):
    - Formed by Senkou Span A and B
    - Acts as dynamic support/resistance
    - Green cloud (A > B) = bullish
    - Red cloud (A < B) = bearish

    Trading Logic:
    - STRONG BUY: Tenkan crosses above Kijun + Price above cloud + Chikou above price
    - BUY: Tenkan crosses above Kijun + Price above cloud
    - STRONG SELL: Tenkan crosses below Kijun + Price below cloud + Chikou below price
    - SELL: Tenkan crosses below Kijun + Price below cloud

    Parameters:
    - tenkan_period: Conversion line period (default: 9)
    - kijun_period: Base line period (default: 26)
    - senkou_b_period: Leading Span B period (default: 52)
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ):
        """
        Initialize Ichimoku Strategy

        Args:
            tenkan_period: Conversion line period (default: 9)
            kijun_period: Base line period (default: 26)
            senkou_b_period: Leading Span B period (default: 52)
        """
        # Validation
        if not isinstance(tenkan_period, int):
            raise TypeError(f"tenkan_period must be int, got {type(tenkan_period).__name__}")
        if tenkan_period < 1:
            raise ValueError(f"tenkan_period must be >= 1, got {tenkan_period}")
        if tenkan_period > 50:
            raise ValueError(f"tenkan_period cannot exceed 50, got {tenkan_period}")

        if not isinstance(kijun_period, int):
            raise TypeError(f"kijun_period must be int, got {type(kijun_period).__name__}")
        if kijun_period < 1:
            raise ValueError(f"kijun_period must be >= 1, got {kijun_period}")
        if kijun_period > 100:
            raise ValueError(f"kijun_period cannot exceed 100, got {kijun_period}")

        if not isinstance(senkou_b_period, int):
            raise TypeError(f"senkou_b_period must be int, got {type(senkou_b_period).__name__}")
        if senkou_b_period < 1:
            raise ValueError(f"senkou_b_period must be >= 1, got {senkou_b_period}")
        if senkou_b_period > 200:
            raise ValueError(f"senkou_b_period cannot exceed 200, got {senkou_b_period}")

        # Logical constraints
        if tenkan_period >= kijun_period:
            raise ValueError(
                f"tenkan_period ({tenkan_period}) should be < kijun_period ({kijun_period}) "
                "for traditional Ichimoku setup"
            )

        if kijun_period >= senkou_b_period:
            raise ValueError(
                f"kijun_period ({kijun_period}) should be < senkou_b_period ({senkou_b_period}) "
                "for traditional Ichimoku setup"
            )

        params = {
            'tenkan_period': tenkan_period,
            'kijun_period': kijun_period,
            'senkou_b_period': senkou_b_period
        }
        super().__init__("Ichimoku Cloud", params)
        logger.debug(
            f"IchimokuStrategy initialized: tenkan={tenkan_period}, "
            f"kijun={kijun_period}, senkou_b={senkou_b_period}"
        )

    def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all Ichimoku components

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dictionary with all Ichimoku lines
        """
        high = df['high'] if 'high' in df.columns else df.get('High', df['close'])
        low = df['low'] if 'low' in df.columns else df.get('Low', df['close'])
        close = df['close'] if 'close' in df.columns else df['Close']

        tenkan_period = self.params['tenkan_period']
        kijun_period = self.params['kijun_period']
        senkou_b_period = self.params['senkou_b_period']

        # 1. Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # 2. Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # 3. Senkou Span A (Leading Span A)
        # Average of Tenkan and Kijun, shifted forward by kijun_period
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

        # 4. Senkou Span B (Leading Span B)
        # Midpoint of 52-period high/low, shifted forward by kijun_period
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

        # 5. Chikou Span (Lagging Span)
        # Close price shifted backward by kijun_period
        chikou_span = close.shift(-kijun_period)

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate Ichimoku trading signals

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (entry_signals, exit_signals)
        """
        close = df['close'] if 'close' in df.columns else df['Close']
        ichimoku = self.calculate_ichimoku(df)

        tenkan = ichimoku['tenkan_sen']
        kijun = ichimoku['kijun_sen']
        senkou_a = ichimoku['senkou_span_a']
        senkou_b = ichimoku['senkou_span_b']
        chikou = ichimoku['chikou_span']

        # Calculate cloud top and bottom (for determining if price is above/below cloud)
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        # Entry Conditions:
        # 1. Tenkan crosses above Kijun (TK cross)
        tk_cross_up = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))

        # 2. Price above cloud
        price_above_cloud = close > cloud_top

        # 3. Chikou Span above price (confirmation)
        # Shift chikou back to align with current period for comparison
        chikou_above_price = chikou.shift(self.params['kijun_period']) > close

        # STRONG BUY: All three conditions
        strong_entries = tk_cross_up & price_above_cloud & chikou_above_price

        # STANDARD BUY: TK cross + price above cloud
        standard_entries = tk_cross_up & price_above_cloud

        # Use standard entries (less strict)
        entries = standard_entries

        # Exit Conditions:
        # 1. Tenkan crosses below Kijun
        tk_cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

        # 2. Price below cloud
        price_below_cloud = close < cloud_bottom

        # STRONG SELL: Both conditions
        strong_exits = tk_cross_down & price_below_cloud

        # STANDARD SELL: TK cross
        standard_exits = tk_cross_down

        # Use standard exits
        exits = standard_exits

        return entries, exits

    def get_description(self) -> str:
        """Get strategy description"""
        return (
            f"Ichimoku Cloud({self.params['tenkan_period']}, "
            f"{self.params['kijun_period']}, {self.params['senkou_b_period']})"
        )

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for optimization"""
        return {
            'tenkan_period': {
                'type': 'int',
                'min': 7,
                'max': 12,
                'default': 9,
                'label': 'Tenkan-sen Period (Conversion Line)'
            },
            'kijun_period': {
                'type': 'int',
                'min': 20,
                'max': 30,
                'default': 26,
                'label': 'Kijun-sen Period (Base Line)'
            },
            'senkou_b_period': {
                'type': 'int',
                'min': 40,
                'max': 60,
                'default': 52,
                'label': 'Senkou Span B Period'
            }
        }

    def get_cloud_color(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine cloud color (bullish/bearish)

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with 1 for bullish (green), -1 for bearish (red)
        """
        ichimoku = self.calculate_ichimoku(df)
        senkou_a = ichimoku['senkou_span_a']
        senkou_b = ichimoku['senkou_span_b']

        # Bullish cloud: Span A > Span B
        # Bearish cloud: Span A < Span B
        cloud_color = pd.Series(index=df.index, dtype=int)
        cloud_color[senkou_a > senkou_b] = 1  # Bullish (green)
        cloud_color[senkou_a < senkou_b] = -1  # Bearish (red)
        cloud_color[senkou_a == senkou_b] = 0  # Neutral

        return cloud_color

    def price_position_relative_to_cloud(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine if price is above, in, or below cloud

        Args:
            df: DataFrame with OHLC data

        Returns:
            Series with 1 (above), 0 (in cloud), -1 (below)
        """
        close = df['close'] if 'close' in df.columns else df['Close']
        ichimoku = self.calculate_ichimoku(df)

        senkou_a = ichimoku['senkou_span_a']
        senkou_b = ichimoku['senkou_span_b']

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        position = pd.Series(index=df.index, dtype=int)
        position[close > cloud_top] = 1  # Above cloud
        position[(close >= cloud_bottom) & (close <= cloud_top)] = 0  # In cloud
        position[close < cloud_bottom] = -1  # Below cloud

        return position
