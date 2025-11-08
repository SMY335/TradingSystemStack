"""
Algorithms for trendline detection.

This module implements various algorithms for detecting trendlines including
regression-based methods and Hough transform.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks as scipy_find_peaks
from dataclasses import dataclass


@dataclass
class TrendlineCandidate:
    """Candidate trendline with its properties."""
    indices: List[int]
    prices: List[float]
    slope: float
    intercept: float
    r_squared: float
    touches: int
    strength: float  # 0-100

    def get_price_at_index(self, index: int) -> float:
        """Calculate price at given index."""
        return self.slope * index + self.intercept


def detect_swing_points(
    prices: np.ndarray,
    order: int = 5,
    mode: str = 'both'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect swing highs and lows in price data.

    Args:
        prices: Array of prices
        order: Number of points on each side for comparison
        mode: 'highs', 'lows', or 'both'

    Returns:
        Tuple of (highs_indices, lows_indices)

    Examples:
        >>> prices = np.array([100, 102, 101, 99, 101, 103, 100])
        >>> highs, lows = detect_swing_points(prices, order=1)
    """
    highs = np.array([])
    lows = np.array([])

    if mode in ['highs', 'both']:
        highs, _ = scipy_find_peaks(prices, distance=order)

    if mode in ['lows', 'both']:
        lows, _ = scipy_find_peaks(-prices, distance=order)

    return highs, lows


def fit_regression_line(
    indices: np.ndarray,
    prices: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit a regression line to points.

    Args:
        indices: X coordinates (bar indices)
        prices: Y coordinates (prices)

    Returns:
        Tuple of (slope, intercept, r_squared)

    Examples:
        >>> indices = np.array([0, 1, 2, 3])
        >>> prices = np.array([100, 101, 102, 103])
        >>> slope, intercept, r2 = fit_regression_line(indices, prices)
    """
    if len(indices) < 2:
        return 0.0, 0.0, 0.0

    slope, intercept, r_value, p_value, std_err = stats.linregress(indices, prices)
    r_squared = r_value ** 2

    return slope, intercept, r_squared


def find_trendline_candidates(
    df: pd.DataFrame,
    price_column: str = 'close',
    min_touches: int = 2,
    tolerance: float = 0.02,
    mode: str = 'both'
) -> List[TrendlineCandidate]:
    """
    Find trendline candidates using swing points.

    Args:
        df: DataFrame with price data
        price_column: Column to use for prices
        min_touches: Minimum number of touches required
        tolerance: Price tolerance for counting touches (as fraction)
        mode: 'support', 'resistance', or 'both'

    Returns:
        List of TrendlineCandidate objects

    Examples:
        >>> candidates = find_trendline_candidates(df, min_touches=3)
    """
    prices = df[price_column].values
    candidates = []

    # Detect swing points
    swing_mode = 'lows' if mode == 'support' else ('highs' if mode == 'resistance' else 'both')
    highs, lows = detect_swing_points(prices, order=5, mode=swing_mode)

    # Process resistance lines (from highs)
    if mode in ['resistance', 'both'] and len(highs) >= 2:
        candidates.extend(_find_lines_from_points(highs, prices, df, 'resistance', min_touches, tolerance))

    # Process support lines (from lows)
    if mode in ['support', 'both'] and len(lows) >= 2:
        candidates.extend(_find_lines_from_points(lows, prices, df, 'support', min_touches, tolerance))

    return candidates


def _find_lines_from_points(
    point_indices: np.ndarray,
    all_prices: np.ndarray,
    df: pd.DataFrame,
    line_type: str,
    min_touches: int,
    tolerance: float
) -> List[TrendlineCandidate]:
    """Helper function to find trendlines from a set of points."""
    candidates = []

    # Try different combinations of points
    for i in range(len(point_indices)):
        for j in range(i + 1, min(i + 10, len(point_indices))):  # Limit combinations
            idx1, idx2 = point_indices[i], point_indices[j]

            # Skip if points are too close
            if (idx2 - idx1) < 10:
                continue

            # Fit line through these two points
            indices = np.array([idx1, idx2])
            prices = all_prices[indices]
            slope, intercept, r_squared = fit_regression_line(indices, prices)

            # Count touches along the entire line
            touches = 0
            touch_indices = []

            for k, price in enumerate(all_prices):
                line_price = slope * k + intercept
                distance = abs(price - line_price)
                threshold = line_price * tolerance

                if distance <= threshold:
                    touches += 1
                    touch_indices.append(k)

            # Check if meets minimum touches
            if touches < min_touches:
                continue

            # Calculate strength (0-100) based on touches, r², and duration
            duration = idx2 - idx1
            strength = min(100, (
                (touches / len(all_prices)) * 100 * 0.4 +
                r_squared * 100 * 0.4 +
                min(duration / 100, 1.0) * 100 * 0.2
            ))

            candidates.append(TrendlineCandidate(
                indices=touch_indices,
                prices=all_prices[touch_indices].tolist(),
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                touches=touches,
                strength=strength
            ))

    return candidates


def merge_similar_trendlines(
    candidates: List[TrendlineCandidate],
    slope_tolerance: float = 0.001,
    intercept_tolerance: float = 1.0
) -> List[TrendlineCandidate]:
    """
    Merge trendlines that are very similar.

    Args:
        candidates: List of trendline candidates
        slope_tolerance: Max slope difference to consider similar
        intercept_tolerance: Max intercept difference to consider similar

    Returns:
        List of merged trendlines

    Examples:
        >>> merged = merge_similar_trendlines(candidates)
    """
    if len(candidates) <= 1:
        return candidates

    # Sort by strength
    sorted_candidates = sorted(candidates, key=lambda x: x.strength, reverse=True)

    merged = []
    used = set()

    for i, candidate in enumerate(sorted_candidates):
        if i in used:
            continue

        # Find similar candidates
        similar = [candidate]

        for j in range(i + 1, len(sorted_candidates)):
            if j in used:
                continue

            other = sorted_candidates[j]

            # Check if similar
            slope_diff = abs(candidate.slope - other.slope)
            intercept_diff = abs(candidate.intercept - other.intercept)

            if slope_diff <= slope_tolerance and intercept_diff <= intercept_tolerance:
                similar.append(other)
                used.add(j)

        # Keep the strongest one
        merged.append(similar[0])

    return merged


def calculate_trendline_strength(
    candidate: TrendlineCandidate,
    total_bars: int
) -> float:
    """
    Calculate strength score for a trendline (0-100).

    Args:
        candidate: Trendline candidate
        total_bars: Total number of bars in data

    Returns:
        Strength score (0-100)

    Examples:
        >>> strength = calculate_trendline_strength(candidate, 100)
    """
    # Touch density (how many bars touch the line)
    touch_density = candidate.touches / total_bars

    # Line quality (R²)
    line_quality = candidate.r_squared

    # Duration (how long the line extends)
    if len(candidate.indices) >= 2:
        duration = candidate.indices[-1] - candidate.indices[0]
        duration_score = min(duration / total_bars, 1.0)
    else:
        duration_score = 0.0

    # Weighted average
    strength = (
        touch_density * 100 * 0.4 +
        line_quality * 100 * 0.4 +
        duration_score * 100 * 0.2
    )

    return min(strength, 100.0)


def filter_trendlines_by_recency(
    candidates: List[TrendlineCandidate],
    current_index: int,
    max_age: int = 100
) -> List[TrendlineCandidate]:
    """
    Filter trendlines to only include recent ones.

    Args:
        candidates: List of trendline candidates
        current_index: Current bar index
        max_age: Maximum age in bars

    Returns:
        Filtered list of recent trendlines

    Examples:
        >>> recent = filter_trendlines_by_recency(candidates, 100, max_age=50)
    """
    recent = []

    for candidate in candidates:
        if len(candidate.indices) > 0:
            # Check if most recent touch is within max_age
            most_recent = max(candidate.indices)
            age = current_index - most_recent

            if age <= max_age:
                recent.append(candidate)

    return recent


def extend_trendline(
    candidate: TrendlineCandidate,
    future_bars: int
) -> np.ndarray:
    """
    Project trendline into the future.

    Args:
        candidate: Trendline to extend
        future_bars: Number of bars to project forward

    Returns:
        Array of projected prices

    Examples:
        >>> projected = extend_trendline(candidate, 20)
    """
    if len(candidate.indices) == 0:
        return np.array([])

    last_index = max(candidate.indices)
    future_indices = np.arange(last_index + 1, last_index + future_bars + 1)
    projected_prices = candidate.slope * future_indices + candidate.intercept

    return projected_prices
