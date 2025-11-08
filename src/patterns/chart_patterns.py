"""
Chart pattern detection algorithms.

This module implements detection logic for various chart patterns including
triangles, head & shoulders, double tops/bottoms, and more.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from .models import (
    ChartPattern,
    TrianglePattern,
    HeadAndShouldersPattern,
    DoubleToppingPattern,
    TripleToppingPattern,
    WedgePattern,
    FlagPattern,
    PennantPattern,
    Point,
    PatternResult,
    PatternType
)
from .geometry import (
    find_peaks,
    find_valleys,
    create_trendline,
    calculate_pattern_confidence,
    calculate_triangle_apex,
    count_trendline_touches,
    smooth_prices
)


class ChartPatternDetector:
    """
    Main class for detecting chart patterns in OHLCV data.

    Examples:
        >>> detector = ChartPatternDetector()
        >>> patterns = detector.detect_all(df)
        >>> triangles = detector.detect_triangles(df)
    """

    def __init__(
        self,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 200,
        min_confidence: float = 0.5,
        smoothing_window: int = 3
    ):
        """
        Initialize pattern detector.

        Args:
            min_pattern_bars: Minimum number of bars for a pattern
            max_pattern_bars: Maximum number of bars for a pattern
            min_confidence: Minimum confidence threshold
            smoothing_window: Window for price smoothing
        """
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window

    def detect_all(
        self,
        df: pd.DataFrame,
        detect_types: Optional[List[PatternType]] = None
    ) -> PatternResult:
        """
        Detect all pattern types in the dataframe.

        Args:
            df: DataFrame with OHLCV data
            detect_types: List of pattern types to detect, or None for all

        Returns:
            PatternResult containing all detected patterns
        """
        all_patterns = []

        # Define detection methods
        detection_methods = {
            'triangle_ascending': lambda: self.detect_triangles(df, 'ascending'),
            'triangle_descending': lambda: self.detect_triangles(df, 'descending'),
            'triangle_symmetrical': lambda: self.detect_triangles(df, 'symmetrical'),
            'head_and_shoulders': lambda: self.detect_head_and_shoulders(df, inverse=False),
            'inverse_head_and_shoulders': lambda: self.detect_head_and_shoulders(df, inverse=True),
            'double_top': lambda: self.detect_double_topping(df, is_top=True),
            'double_bottom': lambda: self.detect_double_topping(df, is_top=False),
        }

        # Run selected detectors
        for pattern_type, method in detection_methods.items():
            if detect_types is None or pattern_type in detect_types:
                try:
                    patterns = method()
                    all_patterns.extend(patterns)
                except Exception as e:
                    # Log error but continue with other patterns
                    print(f"Warning: Failed to detect {pattern_type}: {e}")

        # Filter by confidence
        all_patterns = [p for p in all_patterns if p.confidence >= self.min_confidence]

        return PatternResult(
            patterns=all_patterns,
            detection_time=datetime.now(),
            symbol=df.attrs.get('symbol', 'UNKNOWN'),
            timeframe=df.attrs.get('timeframe', '1d')
        )

    def detect_triangles(
        self,
        df: pd.DataFrame,
        triangle_type: str = 'symmetrical'
    ) -> List[TrianglePattern]:
        """
        Detect triangle patterns (ascending, descending, or symmetrical).

        Args:
            df: DataFrame with OHLCV data
            triangle_type: 'ascending', 'descending', or 'symmetrical'

        Returns:
            List of detected TrianglePattern objects
        """
        patterns = []

        # Smooth prices to reduce noise
        high_smooth = smooth_prices(df['high'].values, self.smoothing_window)
        low_smooth = smooth_prices(df['low'].values, self.smoothing_window)

        # Find peaks and valleys
        peaks = find_peaks(high_smooth, order=5)
        valleys = find_valleys(low_smooth, order=5)

        if len(peaks) < 2 or len(valleys) < 2:
            return patterns

        # Try different combinations of peaks and valleys
        for i in range(len(peaks) - 1):
            for j in range(i + 1, min(i + 5, len(peaks))):  # Limit search window
                # Check if pattern duration is valid
                duration = peaks[j] - peaks[i]
                if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                    continue

                # Get peaks and valleys in range
                range_peaks = peaks[(peaks >= peaks[i]) & (peaks <= peaks[j])]
                range_valleys = valleys[(valleys >= peaks[i]) & (valleys <= peaks[j])]

                if len(range_peaks) < 2 or len(range_valleys) < 2:
                    continue

                # Create trendlines
                try:
                    upper_line = create_trendline(df, range_peaks.tolist(), 'high')
                    lower_line = create_trendline(df, range_valleys.tolist(), 'low')
                except Exception:
                    continue

                # Check triangle type criteria
                is_valid = False
                pattern_type_name = None

                if triangle_type == 'ascending':
                    # Upper flat, lower rising
                    is_valid = abs(upper_line.slope) < 0.01 and lower_line.slope > 0.01
                    pattern_type_name = 'triangle_ascending'

                elif triangle_type == 'descending':
                    # Lower flat, upper falling
                    is_valid = abs(lower_line.slope) < 0.01 and upper_line.slope < -0.01
                    pattern_type_name = 'triangle_descending'

                elif triangle_type == 'symmetrical':
                    # Both converging
                    is_valid = upper_line.slope < -0.01 and lower_line.slope > 0.01
                    pattern_type_name = 'triangle_symmetrical'

                if not is_valid:
                    continue

                # Calculate apex
                apex = calculate_triangle_apex(
                    upper_line.slope, upper_line.intercept,
                    lower_line.slope, lower_line.intercept
                )

                if apex is None or apex < peaks[j]:
                    continue

                # Calculate confidence
                confidence = calculate_pattern_confidence(
                    upper_line.r_squared,
                    lower_line.r_squared,
                    symmetry_score=0.8,
                    touch_count=len(range_peaks) + len(range_valleys)
                )

                # Calculate price range
                prices_in_range = df.loc[peaks[i]:peaks[j], ['high', 'low']]
                price_min = prices_in_range['low'].min()
                price_max = prices_in_range['high'].max()

                # Calculate breakout target (pattern height)
                pattern_height = price_max - price_min
                breakout_target = price_max + pattern_height  # For ascending/symmetrical

                patterns.append(TrianglePattern(
                    pattern_type=pattern_type_name,
                    start_idx=int(peaks[i]),
                    end_idx=int(peaks[j]),
                    start_time=df.index[peaks[i]],
                    end_time=df.index[peaks[j]],
                    confidence=confidence,
                    price_range=(price_min, price_max),
                    breakout_target=breakout_target,
                    breakout_direction='up',
                    upper_trendline=upper_line,
                    lower_trendline=lower_line,
                    apex_index=int(apex),
                    width=int(peaks[j] - peaks[i])
                ))

        return patterns

    def detect_head_and_shoulders(
        self,
        df: pd.DataFrame,
        inverse: bool = False
    ) -> List[HeadAndShouldersPattern]:
        """
        Detect head and shoulders or inverse head and shoulders patterns.

        Args:
            df: DataFrame with OHLCV data
            inverse: If True, detect inverse H&S (bullish), else regular (bearish)

        Returns:
            List of detected HeadAndShouldersPattern objects
        """
        patterns = []

        # Use highs for regular, lows for inverse
        prices = df['low'].values if inverse else df['high'].values
        smoothed = smooth_prices(prices, self.smoothing_window)

        # Find peaks (inverse: valleys)
        if inverse:
            extrema = find_valleys(smoothed, order=5)
        else:
            extrema = find_peaks(smoothed, order=5)

        if len(extrema) < 3:
            return patterns

        # Look for pattern: left shoulder, head, right shoulder
        for i in range(len(extrema) - 2):
            left_idx = extrema[i]
            head_idx = extrema[i + 1]
            right_idx = extrema[i + 2]

            # Check duration
            duration = right_idx - left_idx
            if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                continue

            left_price = prices[left_idx]
            head_price = prices[head_idx]
            right_price = prices[right_idx]

            # Validate H&S structure
            if not inverse:
                # Regular H&S: head should be highest
                if not (head_price > left_price and head_price > right_price):
                    continue
            else:
                # Inverse H&S: head should be lowest
                if not (head_price < left_price and head_price < right_price):
                    continue

            # Create points
            left_shoulder = Point(left_idx, df.index[left_idx], left_price)
            head = Point(head_idx, df.index[head_idx], head_price)
            right_shoulder = Point(right_idx, df.index[right_idx], right_price)

            # Find valleys between peaks to establish neckline (or peaks for inverse)
            if inverse:
                between_extrema = find_peaks(smoothed[left_idx:right_idx+1], order=2)
            else:
                between_extrema = find_valleys(smoothed[left_idx:right_idx+1], order=2)

            if len(between_extrema) < 2:
                continue

            # Create neckline
            neckline_indices = [left_idx + idx for idx in between_extrema[:2]]
            try:
                neckline = create_trendline(
                    df,
                    neckline_indices,
                    'high' if inverse else 'low'
                )
            except Exception:
                continue

            # Calculate symmetry
            left_to_head = abs(head_idx - left_idx)
            head_to_right = abs(right_idx - head_idx)
            symmetry = min(left_to_head, head_to_right) / max(left_to_head, head_to_right)

            # Shoulder height symmetry
            shoulder_symmetry = 1 - abs(left_price - right_price) / max(left_price, right_price)

            # Calculate confidence
            confidence = calculate_pattern_confidence(
                r_squared_upper=0.9,  # Shoulders don't need trendlines
                r_squared_lower=neckline.r_squared,
                symmetry_score=(symmetry + shoulder_symmetry) / 2,
                touch_count=3
            )

            # Price range and target
            price_min = min(left_price, head_price, right_price)
            price_max = max(left_price, head_price, right_price)
            pattern_height = price_max - price_min

            if inverse:
                breakout_target = price_max + pattern_height
                breakout_dir = 'up'
            else:
                breakout_target = price_min - pattern_height
                breakout_dir = 'down'

            patterns.append(HeadAndShouldersPattern(
                pattern_type='inverse_head_and_shoulders' if inverse else 'head_and_shoulders',
                start_idx=int(left_idx),
                end_idx=int(right_idx),
                start_time=df.index[left_idx],
                end_time=df.index[right_idx],
                confidence=confidence,
                price_range=(price_min, price_max),
                breakout_target=breakout_target,
                breakout_direction=breakout_dir,
                left_shoulder=left_shoulder,
                head=head,
                right_shoulder=right_shoulder,
                neckline=neckline
            ))

        return patterns

    def detect_double_topping(
        self,
        df: pd.DataFrame,
        is_top: bool = True
    ) -> List[DoubleToppingPattern]:
        """
        Detect double top or double bottom patterns.

        Args:
            df: DataFrame with OHLCV data
            is_top: If True, detect double tops, else double bottoms

        Returns:
            List of detected DoubleToppingPattern objects
        """
        patterns = []

        # Use highs for tops, lows for bottoms
        prices = df['high'].values if is_top else df['low'].values
        smoothed = smooth_prices(prices, self.smoothing_window)

        # Find peaks/valleys
        if is_top:
            extrema = find_peaks(smoothed, order=5)
        else:
            extrema = find_valleys(smoothed, order=5)

        if len(extrema) < 2:
            return patterns

        # Look for two similar peaks/valleys
        for i in range(len(extrema) - 1):
            first_idx = extrema[i]
            second_idx = extrema[i + 1]

            # Check duration
            duration = second_idx - first_idx
            if duration < self.min_pattern_bars or duration > self.max_pattern_bars:
                continue

            first_price = prices[first_idx]
            second_price = prices[second_idx]

            # Check if peaks/valleys are similar
            price_diff = abs(first_price - second_price)
            avg_price = (first_price + second_price) / 2
            similarity = 1 - (price_diff / avg_price)

            if similarity < 0.95:  # Must be within 5%
                continue

            # Find valley between peaks (or peak between valleys)
            between_prices = smoothed[first_idx:second_idx+1]
            if is_top:
                valley_idx = first_idx + np.argmin(between_prices)
                valley_price = prices[valley_idx]
            else:
                valley_idx = first_idx + np.argmax(between_prices)
                valley_price = prices[valley_idx]

            # Create points
            first_peak = Point(first_idx, df.index[first_idx], first_price)
            valley = Point(valley_idx, df.index[valley_idx], valley_price)
            second_peak = Point(second_idx, df.index[second_idx], second_price)

            # Calculate confidence
            confidence = calculate_pattern_confidence(
                r_squared_upper=0.95,
                r_squared_lower=0.95,
                symmetry_score=similarity,
                touch_count=2
            )

            # Price range and target
            if is_top:
                price_min = valley_price
                price_max = max(first_price, second_price)
                pattern_height = price_max - price_min
                breakout_target = price_min - pattern_height
                breakout_dir = 'down'
            else:
                price_min = min(first_price, second_price)
                price_max = valley_price
                pattern_height = price_max - price_min
                breakout_target = price_max + pattern_height
                breakout_dir = 'up'

            patterns.append(DoubleToppingPattern(
                pattern_type='double_top' if is_top else 'double_bottom',
                start_idx=int(first_idx),
                end_idx=int(second_idx),
                start_time=df.index[first_idx],
                end_time=df.index[second_idx],
                confidence=confidence,
                price_range=(price_min, price_max),
                breakout_target=breakout_target,
                breakout_direction=breakout_dir,
                first_peak=first_peak,
                valley=valley,
                second_peak=second_peak,
                support_resistance_level=valley_price if is_top else valley_price,
                peak_symmetry=similarity
            ))

        return patterns


def detect_patterns(
    df: pd.DataFrame,
    pattern_types: Optional[List[PatternType]] = None,
    min_confidence: float = 0.5
) -> PatternResult:
    """
    Convenience function to detect chart patterns.

    Args:
        df: DataFrame with OHLCV data
        pattern_types: List of pattern types to detect, or None for all
        min_confidence: Minimum confidence threshold

    Returns:
        PatternResult containing detected patterns

    Examples:
        >>> result = detect_patterns(df, pattern_types=['triangle_ascending'])
        >>> print(f"Found {len(result)} patterns")
    """
    detector = ChartPatternDetector(min_confidence=min_confidence)
    return detector.detect_all(df, detect_types=pattern_types)
