"""
Trendline detector for support and resistance levels.

This module provides a high-level interface for detecting support and resistance
trendlines in price data.
"""
import pandas as pd
import numpy as np
from typing import List, Literal, Optional
from dataclasses import dataclass

from .algorithms import (
    find_trendline_candidates,
    merge_similar_trendlines,
    filter_trendlines_by_recency,
    TrendlineCandidate
)


@dataclass
class Trendline:
    """Detected trendline with properties."""
    type: Literal['support', 'resistance']
    slope: float
    intercept: float
    r_squared: float
    touches: int
    strength: float  # 0-100
    start_index: int
    end_index: int
    current_price: float

    def get_price_at_index(self, index: int) -> float:
        """Calculate expected price at given index."""
        return self.slope * index + self.intercept

    def distance_to_price(self, index: int, price: float) -> float:
        """Calculate distance from price to trendline."""
        line_price = self.get_price_at_index(index)
        return price - line_price

    def is_broken(self, index: int, price: float, threshold: float = 0.005) -> bool:
        """
        Check if trendline has been broken.

        Args:
            index: Current bar index
            price: Current price
            threshold: Break threshold (as fraction of line price)

        Returns:
            True if trendline is broken
        """
        line_price = self.get_price_at_index(index)
        distance = abs(price - line_price)
        break_threshold = line_price * threshold

        if self.type == 'support':
            return price < (line_price - break_threshold)
        else:  # resistance
            return price > (line_price + break_threshold)

    def __repr__(self) -> str:
        return (f"Trendline(type={self.type}, slope={self.slope:.4f}, "
                f"strength={self.strength:.1f}, touches={self.touches})")


class TrendlineDetector:
    """
    Detector for support and resistance trendlines.

    Examples:
        >>> detector = TrendlineDetector(min_touches=3, min_strength=50)
        >>> trendlines = detector.detect(df)
        >>> supports = detector.get_support_lines()
        >>> resistances = detector.get_resistance_lines()
    """

    def __init__(
        self,
        min_touches: int = 2,
        min_strength: float = 30.0,
        tolerance: float = 0.02,
        max_age: int = 100,
        merge_similar: bool = True
    ):
        """
        Initialize trendline detector.

        Args:
            min_touches: Minimum number of touches required
            min_strength: Minimum strength score (0-100)
            tolerance: Price tolerance for counting touches
            max_age: Maximum age in bars for recent trendlines
            merge_similar: Whether to merge similar trendlines
        """
        self.min_touches = min_touches
        self.min_strength = min_strength
        self.tolerance = tolerance
        self.max_age = max_age
        self.merge_similar = merge_similar
        self._last_detection = None

    def detect(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        mode: Literal['support', 'resistance', 'both'] = 'both'
    ) -> List[Trendline]:
        """
        Detect trendlines in price data.

        Args:
            df: DataFrame with OHLCV data
            price_column: Column to use for detection
            mode: Type of trendlines to detect

        Returns:
            List of detected Trendline objects

        Examples:
            >>> trendlines = detector.detect(df, mode='both')
            >>> print(f"Found {len(trendlines)} trendlines")
        """
        # Find candidates
        candidates = find_trendline_candidates(
            df,
            price_column=price_column,
            min_touches=self.min_touches,
            tolerance=self.tolerance,
            mode=mode
        )

        # Filter by strength
        candidates = [c for c in candidates if c.strength >= self.min_strength]

        # Filter by recency
        current_index = len(df) - 1
        candidates = filter_trendlines_by_recency(
            candidates,
            current_index,
            max_age=self.max_age
        )

        # Merge similar trendlines
        if self.merge_similar:
            candidates = merge_similar_trendlines(candidates)

        # Convert to Trendline objects
        trendlines = []
        current_price = df[price_column].iloc[-1]

        for candidate in candidates:
            if len(candidate.indices) == 0:
                continue

            # Determine type based on slope and position
            line_type = self._determine_type(candidate, df[price_column].values)

            trendlines.append(Trendline(
                type=line_type,
                slope=candidate.slope,
                intercept=candidate.intercept,
                r_squared=candidate.r_squared,
                touches=candidate.touches,
                strength=candidate.strength,
                start_index=min(candidate.indices),
                end_index=max(candidate.indices),
                current_price=current_price
            ))

        # Store for later queries
        self._last_detection = trendlines

        # Sort by strength
        trendlines.sort(key=lambda x: x.strength, reverse=True)

        return trendlines

    def get_support_lines(self) -> List[Trendline]:
        """Get only support trendlines from last detection."""
        if self._last_detection is None:
            return []
        return [t for t in self._last_detection if t.type == 'support']

    def get_resistance_lines(self) -> List[Trendline]:
        """Get only resistance trendlines from last detection."""
        if self._last_detection is None:
            return []
        return [t for t in self._last_detection if t.type == 'resistance']

    def get_nearest_support(self, current_index: int, current_price: float) -> Optional[Trendline]:
        """
        Find nearest support line below current price.

        Args:
            current_index: Current bar index
            current_price: Current price

        Returns:
            Nearest support Trendline or None
        """
        supports = self.get_support_lines()

        nearest = None
        min_distance = float('inf')

        for support in supports:
            line_price = support.get_price_at_index(current_index)

            if line_price < current_price:
                distance = current_price - line_price

                if distance < min_distance:
                    min_distance = distance
                    nearest = support

        return nearest

    def get_nearest_resistance(self, current_index: int, current_price: float) -> Optional[Trendline]:
        """
        Find nearest resistance line above current price.

        Args:
            current_index: Current bar index
            current_price: Current price

        Returns:
            Nearest resistance Trendline or None
        """
        resistances = self.get_resistance_lines()

        nearest = None
        min_distance = float('inf')

        for resistance in resistances:
            line_price = resistance.get_price_at_index(current_index)

            if line_price > current_price:
                distance = line_price - current_price

                if distance < min_distance:
                    min_distance = distance
                    nearest = resistance

        return nearest

    @staticmethod
    def _determine_type(candidate: TrendlineCandidate, all_prices: np.ndarray) -> str:
        """Determine if trendline is support or resistance."""
        # Count how many prices are above vs below the line
        above = 0
        below = 0

        for i, price in enumerate(all_prices):
            line_price = candidate.slope * i + candidate.intercept

            if price > line_price:
                above += 1
            elif price < line_price:
                below += 1

        # If most prices are above, it's support
        # If most prices are below, it's resistance
        return 'support' if above > below else 'resistance'


def detect_support_resistance(
    df: pd.DataFrame,
    min_touches: int = 2,
    min_strength: float = 30.0
) -> List[Trendline]:
    """
    Convenience function to detect support and resistance lines.

    Args:
        df: DataFrame with OHLCV data
        min_touches: Minimum number of touches
        min_strength: Minimum strength score

    Returns:
        List of Trendline objects

    Examples:
        >>> trendlines = detect_support_resistance(df)
        >>> supports = [t for t in trendlines if t.type == 'support']
    """
    detector = TrendlineDetector(
        min_touches=min_touches,
        min_strength=min_strength
    )
    return detector.detect(df)
