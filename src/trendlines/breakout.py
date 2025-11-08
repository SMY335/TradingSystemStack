"""
Breakout detection for trendlines.

This module provides functionality to detect when price breaks through
support or resistance levels.
"""
import pandas as pd
import numpy as np
from typing import List, Literal, Optional
from dataclasses import dataclass
from datetime import datetime

from .detector import Trendline


@dataclass
class Breakout:
    """Detected breakout event."""
    trendline_type: Literal['support', 'resistance']
    breakout_direction: Literal['up', 'down']
    breakout_index: int
    breakout_time: datetime
    breakout_price: float
    line_price: float
    break_strength: float  # How strong the break (as % of line price)
    volume_surge: Optional[float] = None  # Volume increase ratio

    def is_bullish(self) -> bool:
        """Check if breakout is bullish."""
        return (self.trendline_type == 'resistance' and self.breakout_direction == 'up') or \
               (self.trendline_type == 'support' and self.breakout_direction == 'down')

    def is_bearish(self) -> bool:
        """Check if breakout is bearish."""
        return not self.is_bullish()

    def __repr__(self) -> str:
        return (f"Breakout({self.trendline_type} break {self.breakout_direction}, "
                f"strength={self.break_strength:.2%})")


class BreakoutDetector:
    """
    Detector for trendline breakouts.

    Examples:
        >>> detector = BreakoutDetector(min_break_pct=0.01)
        >>> breakouts = detector.detect_breakouts(df, trendlines)
        >>> bullish = [b for b in breakouts if b.is_bullish()]
    """

    def __init__(
        self,
        min_break_pct: float = 0.005,
        require_close_beyond: bool = True,
        min_volume_surge: Optional[float] = None,
        confirmation_bars: int = 1
    ):
        """
        Initialize breakout detector.

        Args:
            min_break_pct: Minimum break percentage (as fraction)
            require_close_beyond: Require close beyond line, not just wick
            min_volume_surge: Minimum volume increase ratio (e.g., 1.5 = 50% increase)
            confirmation_bars: Number of bars to confirm breakout
        """
        self.min_break_pct = min_break_pct
        self.require_close_beyond = require_close_beyond
        self.min_volume_surge = min_volume_surge
        self.confirmation_bars = confirmation_bars

    def detect_breakouts(
        self,
        df: pd.DataFrame,
        trendlines: List[Trendline],
        start_index: Optional[int] = None
    ) -> List[Breakout]:
        """
        Detect breakouts of trendlines in price data.

        Args:
            df: DataFrame with OHLCV data
            trendlines: List of trendlines to check for breakouts
            start_index: Index to start checking from (None = start of data)

        Returns:
            List of Breakout objects

        Examples:
            >>> breakouts = detector.detect_breakouts(df, trendlines)
        """
        breakouts = []

        if start_index is None:
            start_index = max(t.start_index for t in trendlines) if trendlines else 0

        for i in range(start_index, len(df)):
            for trendline in trendlines:
                # Skip if before trendline start
                if i < trendline.start_index:
                    continue

                # Check for breakout at this bar
                breakout = self._check_breakout(df, trendline, i)

                if breakout is not None:
                    breakouts.append(breakout)

        return breakouts

    def _check_breakout(
        self,
        df: pd.DataFrame,
        trendline: Trendline,
        index: int
    ) -> Optional[Breakout]:
        """Check if a breakout occurred at given index."""
        if index >= len(df):
            return None

        bar = df.iloc[index]
        line_price = trendline.get_price_at_index(index)

        # Determine which price to use
        price_to_check = bar['close'] if self.require_close_beyond else (
            bar['high'] if trendline.type == 'resistance' else bar['low']
        )

        # Calculate break distance
        if trendline.type == 'resistance':
            break_distance = price_to_check - line_price
            breakout_direction = 'up'
        else:  # support
            break_distance = line_price - price_to_check
            breakout_direction = 'down'

        # Check if break threshold met
        break_pct = abs(break_distance) / line_price

        if break_pct < self.min_break_pct:
            return None

        # Check if break is in correct direction
        if (trendline.type == 'resistance' and break_distance <= 0) or \
           (trendline.type == 'support' and break_distance <= 0):
            return None

        # Check volume surge if required
        volume_surge = None
        if self.min_volume_surge is not None and 'volume' in df.columns:
            if index > 20:  # Need historical data
                avg_volume = df['volume'].iloc[index-20:index].mean()
                current_volume = bar['volume']

                if avg_volume > 0:
                    volume_surge = current_volume / avg_volume

                    if volume_surge < self.min_volume_surge:
                        return None  # Volume requirement not met

        # Check confirmation if required
        if self.confirmation_bars > 0 and index + self.confirmation_bars < len(df):
            confirmed = self._check_confirmation(df, trendline, index)
            if not confirmed:
                return None

        # Create breakout event
        return Breakout(
            trendline_type=trendline.type,
            breakout_direction=breakout_direction,
            breakout_index=index,
            breakout_time=df.index[index],
            breakout_price=price_to_check,
            line_price=line_price,
            break_strength=break_pct,
            volume_surge=volume_surge
        )

    def _check_confirmation(
        self,
        df: pd.DataFrame,
        trendline: Trendline,
        breakout_index: int
    ) -> bool:
        """Check if breakout is confirmed by subsequent bars."""
        for i in range(1, self.confirmation_bars + 1):
            confirm_index = breakout_index + i

            if confirm_index >= len(df):
                return False

            bar = df.iloc[confirm_index]
            line_price = trendline.get_price_at_index(confirm_index)

            # For resistance breakout, close should stay above line
            # For support breakout, close should stay below line
            if trendline.type == 'resistance':
                if bar['close'] < line_price:
                    return False
            else:  # support
                if bar['close'] > line_price:
                    return False

        return True

    def detect_retests(
        self,
        df: pd.DataFrame,
        breakouts: List[Breakout],
        max_bars_after: int = 20,
        retest_tolerance: float = 0.01
    ) -> List[dict]:
        """
        Detect retests of broken trendlines.

        Args:
            df: DataFrame with OHLCV data
            breakouts: List of breakout events
            max_bars_after: Maximum bars after breakout to look for retest
            retest_tolerance: Price tolerance for retest (as fraction)

        Returns:
            List of retest events

        Examples:
            >>> retests = detector.detect_retests(df, breakouts)
        """
        retests = []

        for breakout in breakouts:
            start_check = breakout.breakout_index + 1
            end_check = min(breakout.breakout_index + max_bars_after, len(df))

            for i in range(start_check, end_check):
                bar = df.iloc[i]

                # Calculate distance to broken line
                line_price = breakout.line_price
                distance = abs(bar['low'] - line_price) / line_price

                # Check if price came back to test the line
                if distance <= retest_tolerance:
                    # For resistance -> support flip
                    if breakout.trendline_type == 'resistance' and breakout.breakout_direction == 'up':
                        # Price should touch from above (low near line)
                        if bar['low'] <= line_price * (1 + retest_tolerance) and \
                           bar['close'] > line_price:
                            retests.append({
                                'breakout_index': breakout.breakout_index,
                                'retest_index': i,
                                'retest_time': df.index[i],
                                'retest_price': bar['low'],
                                'line_price': line_price,
                                'bars_after': i - breakout.breakout_index,
                                'held': bar['close'] > line_price  # Did it hold as support?
                            })
                            break  # Only count first retest

                    # For support -> resistance flip
                    elif breakout.trendline_type == 'support' and breakout.breakout_direction == 'down':
                        # Price should touch from below (high near line)
                        if bar['high'] >= line_price * (1 - retest_tolerance) and \
                           bar['close'] < line_price:
                            retests.append({
                                'breakout_index': breakout.breakout_index,
                                'retest_index': i,
                                'retest_time': df.index[i],
                                'retest_price': bar['high'],
                                'line_price': line_price,
                                'bars_after': i - breakout.breakout_index,
                                'held': bar['close'] < line_price  # Did it hold as resistance?
                            })
                            break

        return retests


def detect_breakouts(
    df: pd.DataFrame,
    trendlines: List[Trendline],
    min_break_pct: float = 0.005,
    require_volume_surge: bool = False
) -> List[Breakout]:
    """
    Convenience function to detect trendline breakouts.

    Args:
        df: DataFrame with OHLCV data
        trendlines: List of trendlines
        min_break_pct: Minimum break percentage
        require_volume_surge: Whether to require volume confirmation

    Returns:
        List of Breakout objects

    Examples:
        >>> breakouts = detect_breakouts(df, trendlines)
        >>> bullish_breaks = [b for b in breakouts if b.is_bullish()]
    """
    detector = BreakoutDetector(
        min_break_pct=min_break_pct,
        min_volume_surge=1.5 if require_volume_surge else None
    )
    return detector.detect_breakouts(df, trendlines)
