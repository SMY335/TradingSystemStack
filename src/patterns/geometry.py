"""
Geometric calculations for pattern detection.

This module provides utilities for finding peaks, valleys, trendlines,
and performing geometric analysis on price data.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from scipy.signal import argrelextrema

from .models import Point, TrendLine


def find_peaks(prices: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find local maxima (peaks) in price data.

    Args:
        prices: Array of prices
        order: How many points on each side to use for comparison

    Returns:
        Array of indices where peaks occur

    Examples:
        >>> prices = np.array([1, 2, 3, 2, 1, 2, 4, 3, 2])
        >>> peaks = find_peaks(prices, order=1)
        >>> peaks
        array([2, 6])
    """
    peaks = argrelextrema(prices, np.greater, order=order)[0]
    return peaks


def find_valleys(prices: np.ndarray, order: int = 5) -> np.ndarray:
    """
    Find local minima (valleys) in price data.

    Args:
        prices: Array of prices
        order: How many points on each side to use for comparison

    Returns:
        Array of indices where valleys occur

    Examples:
        >>> prices = np.array([3, 2, 1, 2, 3, 2, 1, 2, 3])
        >>> valleys = find_valleys(prices, order=1)
        >>> valleys
        array([2, 6])
    """
    valleys = argrelextrema(prices, np.less, order=order)[0]
    return valleys


def fit_trendline(indices: np.ndarray, prices: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit a linear trendline to given points using linear regression.

    Args:
        indices: Array of x-coordinates (bar indices)
        prices: Array of y-coordinates (prices)

    Returns:
        Tuple of (slope, intercept, r_squared)

    Examples:
        >>> indices = np.array([0, 1, 2, 3, 4])
        >>> prices = np.array([100, 101, 102, 103, 104])
        >>> slope, intercept, r2 = fit_trendline(indices, prices)
        >>> slope
        1.0
    """
    if len(indices) < 2:
        return 0.0, 0.0, 0.0

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(indices, prices)
    r_squared = r_value ** 2

    return slope, intercept, r_squared


def create_trendline(
    df: pd.DataFrame,
    point_indices: List[int],
    price_column: str = 'close'
) -> TrendLine:
    """
    Create a TrendLine object from DataFrame points.

    Args:
        df: DataFrame with price data
        point_indices: List of indices to use for trendline
        price_column: Column name for prices

    Returns:
        TrendLine object

    Examples:
        >>> df = pd.DataFrame({'close': [100, 101, 102, 103]})
        >>> trendline = create_trendline(df, [0, 1, 2, 3])
    """
    if len(point_indices) < 2:
        raise ValueError("Need at least 2 points to create a trendline")

    points = []
    indices_array = np.array(point_indices)
    prices_array = df[price_column].iloc[point_indices].values

    # Create Point objects
    for idx in point_indices:
        points.append(Point(
            index=idx,
            timestamp=df.index[idx],
            price=df[price_column].iloc[idx]
        ))

    # Fit trendline
    slope, intercept, r_squared = fit_trendline(indices_array, prices_array)

    return TrendLine(
        points=points,
        slope=slope,
        intercept=intercept,
        r_squared=r_squared
    )


def find_support_resistance(
    prices: np.ndarray,
    tolerance: float = 0.02
) -> List[Tuple[float, int]]:
    """
    Find support and resistance levels based on price clusters.

    Args:
        prices: Array of prices
        tolerance: Price tolerance for clustering (as fraction)

    Returns:
        List of (price_level, touch_count) tuples

    Examples:
        >>> prices = np.array([100, 101, 100, 99, 100, 101, 100])
        >>> levels = find_support_resistance(prices, tolerance=0.02)
    """
    # Find peaks and valleys
    peaks = find_peaks(prices, order=3)
    valleys = find_valleys(prices, order=3)

    # Combine all significant prices
    significant_prices = np.concatenate([
        prices[peaks],
        prices[valleys]
    ])

    if len(significant_prices) == 0:
        return []

    # Cluster similar prices
    levels = []
    used = np.zeros(len(significant_prices), dtype=bool)

    for i, price in enumerate(significant_prices):
        if used[i]:
            continue

        # Find all prices within tolerance
        mask = np.abs(significant_prices - price) / price <= tolerance
        cluster = significant_prices[mask]
        used[mask] = True

        # Calculate level as mean of cluster
        level = np.mean(cluster)
        count = len(cluster)

        levels.append((level, count))

    # Sort by touch count (descending)
    levels.sort(key=lambda x: x[1], reverse=True)

    return levels


def calculate_angle(slope: float) -> float:
    """
    Convert slope to angle in degrees.

    Args:
        slope: Line slope

    Returns:
        Angle in degrees

    Examples:
        >>> angle = calculate_angle(1.0)  # 45 degrees
        >>> angle
        45.0
    """
    return np.degrees(np.arctan(slope))


def lines_intersect(
    slope1: float,
    intercept1: float,
    slope2: float,
    intercept2: float,
    x_min: float = 0,
    x_max: float = 1000
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines.

    Args:
        slope1: Slope of first line
        intercept1: Intercept of first line
        slope2: Slope of second line
        intercept2: Intercept of second line
        x_min: Minimum x value to consider
        x_max: Maximum x value to consider

    Returns:
        (x, y) coordinates of intersection, or None if lines are parallel

    Examples:
        >>> # Lines: y = x + 1 and y = -x + 3
        >>> intersection = lines_intersect(1, 1, -1, 3)
        >>> intersection
        (1.0, 2.0)
    """
    # Check if lines are parallel
    if abs(slope1 - slope2) < 1e-10:
        return None

    # Calculate intersection
    x = (intercept2 - intercept1) / (slope1 - slope2)
    y = slope1 * x + intercept1

    # Check if intersection is within bounds
    if x_min <= x <= x_max:
        return (x, y)

    return None


def calculate_pattern_confidence(
    r_squared_upper: float,
    r_squared_lower: float,
    symmetry_score: float = 1.0,
    touch_count: int = 3
) -> float:
    """
    Calculate overall confidence score for a pattern.

    Args:
        r_squared_upper: R-squared of upper trendline
        r_squared_lower: R-squared of lower trendline
        symmetry_score: Symmetry score (0-1)
        touch_count: Number of touches on trendlines

    Returns:
        Confidence score (0-1)

    Examples:
        >>> confidence = calculate_pattern_confidence(0.95, 0.90, 0.85, 4)
        >>> confidence > 0.8
        True
    """
    # Average trendline fit quality
    avg_fit = (r_squared_upper + r_squared_lower) / 2

    # Touch bonus (more touches = more confidence)
    touch_bonus = min(touch_count / 5, 1.0)  # Cap at 5 touches

    # Weighted average
    confidence = (
        0.5 * avg_fit +
        0.3 * symmetry_score +
        0.2 * touch_bonus
    )

    return np.clip(confidence, 0.0, 1.0)


def is_price_near_line(
    price: float,
    index: float,
    slope: float,
    intercept: float,
    tolerance: float = 0.02
) -> bool:
    """
    Check if a price is near a trendline.

    Args:
        price: Price to check
        index: X-coordinate (bar index)
        slope: Line slope
        intercept: Line intercept
        tolerance: Distance tolerance (as fraction of price)

    Returns:
        True if price is near line

    Examples:
        >>> # Line: y = 2x + 100
        >>> is_price_near_line(102, 1, 2, 100, tolerance=0.01)
        True
    """
    line_price = slope * index + intercept
    distance = abs(price - line_price)
    threshold = line_price * tolerance

    return distance <= threshold


def count_trendline_touches(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    price_column: str = 'close',
    tolerance: float = 0.02
) -> int:
    """
    Count how many times price touches a trendline.

    Args:
        df: DataFrame with price data
        slope: Trendline slope
        intercept: Trendline intercept
        price_column: Column to use for prices
        tolerance: Touch tolerance (as fraction)

    Returns:
        Number of touches

    Examples:
        >>> df = pd.DataFrame({'close': [100, 102, 104, 106]})
        >>> touches = count_trendline_touches(df, 2, 100, tolerance=0.01)
        >>> touches
        4
    """
    touches = 0

    for i, price in enumerate(df[price_column]):
        if is_price_near_line(price, i, slope, intercept, tolerance):
            touches += 1

    return touches


def calculate_triangle_apex(
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float
) -> Optional[int]:
    """
    Calculate where triangle apex (convergence point) occurs.

    Args:
        upper_slope: Upper trendline slope
        upper_intercept: Upper trendline intercept
        lower_slope: Lower trendline slope
        lower_intercept: Lower trendline intercept

    Returns:
        Index where lines converge, or None if parallel

    Examples:
        >>> apex = calculate_triangle_apex(0.5, 100, -0.5, 90)
        >>> apex
        10.0
    """
    intersection = lines_intersect(
        upper_slope, upper_intercept,
        lower_slope, lower_intercept
    )

    if intersection is None:
        return None

    return int(intersection[0])


def smooth_prices(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth prices using moving average to reduce noise.

    Args:
        prices: Array of prices
        window: Smoothing window size

    Returns:
        Smoothed prices

    Examples:
        >>> prices = np.array([100, 102, 98, 101, 99, 103])
        >>> smoothed = smooth_prices(prices, window=3)
    """
    if len(prices) < window:
        return prices

    return pd.Series(prices).rolling(window, center=True, min_periods=1).mean().values
