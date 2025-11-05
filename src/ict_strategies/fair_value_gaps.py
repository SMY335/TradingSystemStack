"""
Fair Value Gap Detector - ICT Methodology

Fair Value Gaps (FVG) are price imbalances left by rapid market movements.
Price tends to return to fill these gaps, creating trading opportunities.
"""

import pandas as pd
from typing import List
from dataclasses import dataclass
from enum import Enum


class FVGType(Enum):
    BULLISH = "bullish"  # Price should return to fill the gap
    BEARISH = "bearish"


@dataclass
class FairValueGap:
    type: FVGType
    gap_start: float
    gap_end: float
    candle_idx: int
    filled: bool = False
    fill_pct: float = 0.0


class FairValueGapDetector:
    """
    Fair Value Gap (FVG) = Imbalance in price (gap) left by rapid movement
    
    Bullish FVG: Low[i] > High[i-2] (gap between 2 candles moving up)
    Bearish FVG: High[i] < Low[i-2] (gap between 2 candles moving down)
    
    Price tends to return to fill these gaps = trading opportunity
    """
    
    def __init__(self, min_gap_pct: float = 0.001):
        """
        Args:
            min_gap_pct: Minimum gap size as percentage of price
        """
        self.min_gap_pct = min_gap_pct
    
    def detect(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detect all Fair Value Gaps"""
        gaps = []
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                gap_size = df.iloc[i]['low'] - df.iloc[i-2]['high']
                gap_pct = gap_size / df.iloc[i-2]['high']
                
                if gap_pct >= self.min_gap_pct:
                    gaps.append(FairValueGap(
                        type=FVGType.BULLISH,
                        gap_start=df.iloc[i-2]['high'],
                        gap_end=df.iloc[i]['low'],
                        candle_idx=i
                    ))
            
            # Bearish FVG
            if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                gap_size = df.iloc[i-2]['low'] - df.iloc[i]['high']
                gap_pct = gap_size / df.iloc[i]['high']
                
                if gap_pct >= self.min_gap_pct:
                    gaps.append(FairValueGap(
                        type=FVGType.BEARISH,
                        gap_start=df.iloc[i]['high'],
                        gap_end=df.iloc[i-2]['low'],
                        candle_idx=i
                    ))
        
        return gaps
    
    def update_gaps(self, gaps: List[FairValueGap], df: pd.DataFrame, current_idx: int) -> List[FairValueGap]:
        """Update the status of gaps (filled or not)"""
        for gap in gaps:
            if gap.filled or gap.candle_idx >= current_idx:
                continue
            
            # Check if price has revisited the gap
            price_since = df.iloc[gap.candle_idx:current_idx+1]
            
            if gap.type == FVGType.BULLISH:
                # Price must go down into the gap
                lowest = price_since['low'].min()
                if lowest <= gap.gap_end:
                    gap.fill_pct = min(100, ((gap.gap_end - lowest) / (gap.gap_end - gap.gap_start)) * 100)
                    if lowest <= gap.gap_start:
                        gap.filled = True
            
            elif gap.type == FVGType.BEARISH:
                # Price must go up into the gap
                highest = price_since['high'].max()
                if highest >= gap.gap_start:
                    gap.fill_pct = min(100, ((highest - gap.gap_start) / (gap.gap_end - gap.gap_start)) * 100)
                    if highest >= gap.gap_end:
                        gap.filled = True
        
        return gaps
