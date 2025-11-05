"""
Order Block Detector - ICT Methodology

Order Blocks are zones where institutions have placed large orders.
They represent areas of significant supply/demand imbalance.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OrderBlockType(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class OrderBlock:
    type: OrderBlockType
    start_idx: int
    end_idx: int
    high: float
    low: float
    open: float
    close: float
    volume: float
    strength: float  # Score 0-100
    mitigated: bool = False


class OrderBlockDetector:
    """
    Detects Order Blocks according to ICT methodology
    
    Order Block = Zone where institutions placed massive orders
    - Bullish OB: Last bearish candle before strong bullish move
    - Bearish OB: Last bullish candle before strong bearish move
    """
    
    def __init__(self, min_body_pct: float = 0.6, lookback: int = 20):
        """
        Args:
            min_body_pct: Minimum body percentage of candle range
            lookback: Lookback period for calculations
        """
        self.min_body_pct = min_body_pct
        self.lookback = lookback
    
    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Detect all order blocks in the dataframe"""
        order_blocks = []
        
        for i in range(self.lookback, len(df) - 3):
            # Bullish Order Block
            if self._is_bullish_ob(df, i):
                ob = self._create_order_block(df, i, OrderBlockType.BULLISH)
                order_blocks.append(ob)
            
            # Bearish Order Block
            if self._is_bearish_ob(df, i):
                ob = self._create_order_block(df, i, OrderBlockType.BEARISH)
                order_blocks.append(ob)
        
        return order_blocks
    
    def _is_bullish_ob(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Bullish OB = Last bearish candle before bullish breakout
        """
        current = df.iloc[idx]
        next_candles = df.iloc[idx+1:idx+4]
        
        # Current must be bearish
        if current['close'] >= current['open']:
            return False
        
        # Body must be significant
        body = abs(current['close'] - current['open'])
        candle_range = current['high'] - current['low']
        if candle_range == 0 or body / candle_range < self.min_body_pct:
            return False
        
        # Next 3 candles must break the high
        if not any(next_candles['high'] > current['high']):
            return False
        
        # Strong bullish move (>2% in 3 candles)
        price_change = (next_candles.iloc[-1]['close'] - current['close']) / current['close']
        return price_change > 0.02
    
    def _is_bearish_ob(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Bearish OB = Last bullish candle before bearish breakout
        """
        current = df.iloc[idx]
        next_candles = df.iloc[idx+1:idx+4]
        
        # Current must be bullish
        if current['close'] <= current['open']:
            return False
        
        # Body must be significant
        body = abs(current['close'] - current['open'])
        candle_range = current['high'] - current['low']
        if candle_range == 0 or body / candle_range < self.min_body_pct:
            return False
        
        # Next 3 candles must break the low
        if not any(next_candles['low'] < current['low']):
            return False
        
        # Strong bearish move (<-2% in 3 candles)
        price_change = (next_candles.iloc[-1]['close'] - current['close']) / current['close']
        return price_change < -0.02
    
    def _create_order_block(self, df: pd.DataFrame, idx: int, ob_type: OrderBlockType) -> OrderBlock:
        """Create an OrderBlock with scoring"""
        candle = df.iloc[idx]
        
        # Calculate strength based on volume and size
        volume_rank = self._calculate_volume_rank(df, idx)
        size_rank = self._calculate_size_rank(df, idx)
        strength = (volume_rank + size_rank) / 2
        
        return OrderBlock(
            type=ob_type,
            start_idx=idx,
            end_idx=idx,
            high=candle['high'],
            low=candle['low'],
            open=candle['open'],
            close=candle['close'],
            volume=candle['volume'],
            strength=strength
        )
    
    def _calculate_volume_rank(self, df: pd.DataFrame, idx: int) -> float:
        """Volume percentile over lookback period"""
        lookback_volumes = df.iloc[max(0, idx-self.lookback):idx]['volume']
        if len(lookback_volumes) == 0:
            return 50.0
        current_volume = df.iloc[idx]['volume']
        percentile = (lookback_volumes < current_volume).sum() / len(lookback_volumes)
        return percentile * 100
    
    def _calculate_size_rank(self, df: pd.DataFrame, idx: int) -> float:
        """Candle size percentile over lookback period"""
        lookback_ranges = (df.iloc[max(0, idx-self.lookback):idx]['high'] - 
                          df.iloc[max(0, idx-self.lookback):idx]['low'])
        if len(lookback_ranges) == 0:
            return 50.0
        current_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
        percentile = (lookback_ranges < current_range).sum() / len(lookback_ranges)
        return percentile * 100
