"""
Liquidity Pool Detector - ICT Methodology

Liquidity Pools are zones where many stop losses are placed.
Smart money "sweeps" these pools before moving in the opposite direction.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LiquidityPool:
    price_level: float
    pool_type: str  # "buy_side" (resistance) or "sell_side" (support)
    touches: int  # Number of times tested
    volume_at_level: float
    strength: float  # Score 0-100
    swept: bool = False  # If the level was broken (stop hunted)


class LiquidityPoolDetector:
    """
    Liquidity Pools = Zones where many stop losses are placed
    
    - Buy-side liquidity: Above swing highs (stops of shorts)
    - Sell-side liquidity: Below swing lows (stops of longs)
    
    Smart money "sweeps" these pools before moving in the opposite direction
    """
    
    def __init__(self, lookback: int = 50, touch_tolerance: float = 0.001):
        """
        Args:
            lookback: Lookback period for swing detection
            touch_tolerance: Price tolerance for counting touches
        """
        self.lookback = lookback
        self.touch_tolerance = touch_tolerance
    
    def detect_pools(self, df: pd.DataFrame) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
        """Detect buy-side and sell-side liquidity"""
        buy_side_pools = self._detect_buy_side(df)
        sell_side_pools = self._detect_sell_side(df)
        
        return buy_side_pools, sell_side_pools
    
    def _detect_buy_side(self, df: pd.DataFrame) -> List[LiquidityPool]:
        """Buy-side liquidity = Resistances (swing highs)"""
        pools = []
        
        for i in range(self.lookback, len(df) - self.lookback):
            # Find swing high
            if self._is_swing_high(df, i):
                high = df.iloc[i]['high']
                touches = self._count_touches(df, high, i, 'high')
                volume = df.iloc[i]['volume']
                
                if touches >= 2:  # At least 2 touches
                    strength = min(100, touches * 20 + (volume / df['volume'].mean()) * 10)
                    
                    pools.append(LiquidityPool(
                        price_level=high,
                        pool_type="buy_side",
                        touches=touches,
                        volume_at_level=volume,
                        strength=strength
                    ))
        
        return pools
    
    def _detect_sell_side(self, df: pd.DataFrame) -> List[LiquidityPool]:
        """Sell-side liquidity = Supports (swing lows)"""
        pools = []
        
        for i in range(self.lookback, len(df) - self.lookback):
            # Find swing low
            if self._is_swing_low(df, i):
                low = df.iloc[i]['low']
                touches = self._count_touches(df, low, i, 'low')
                volume = df.iloc[i]['volume']
                
                if touches >= 2:
                    strength = min(100, touches * 20 + (volume / df['volume'].mean()) * 10)
                    
                    pools.append(LiquidityPool(
                        price_level=low,
                        pool_type="sell_side",
                        touches=touches,
                        volume_at_level=volume,
                        strength=strength
                    ))
        
        return pools
    
    def _is_swing_high(self, df: pd.DataFrame, idx: int, left_bars: int = 5, right_bars: int = 5) -> bool:
        """Check if this is a swing high"""
        if idx < left_bars or idx >= len(df) - right_bars:
            return False
        
        high = df.iloc[idx]['high']
        left = df.iloc[idx-left_bars:idx]['high']
        right = df.iloc[idx+1:idx+right_bars+1]['high']
        
        return (high > left.max()) and (high > right.max())
    
    def _is_swing_low(self, df: pd.DataFrame, idx: int, left_bars: int = 5, right_bars: int = 5) -> bool:
        """Check if this is a swing low"""
        if idx < left_bars or idx >= len(df) - right_bars:
            return False
        
        low = df.iloc[idx]['low']
        left = df.iloc[idx-left_bars:idx]['low']
        right = df.iloc[idx+1:idx+right_bars+1]['low']
        
        return (low < left.min()) and (low < right.min())
    
    def _count_touches(self, df: pd.DataFrame, level: float, center_idx: int, col: str) -> int:
        """Count how many times the level was touched"""
        window = df.iloc[max(0, center_idx-self.lookback):min(len(df), center_idx+self.lookback)]
        
        tolerance = level * self.touch_tolerance
        touches = 0
        
        for price in window[col]:
            if abs(price - level) <= tolerance:
                touches += 1
        
        return touches
