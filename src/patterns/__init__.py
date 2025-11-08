"""
Chart Pattern Detection Module.

This module provides comprehensive chart pattern detection capabilities including:
- Triangles (ascending, descending, symmetrical)
- Head and shoulders (regular and inverse)
- Double tops and bottoms
- Triple tops and bottoms
- Wedges (rising and falling)
- Flags (bullish and bearish)
- Pennants

Examples:
    >>> from src.patterns import detect_patterns, ChartPatternDetector
    >>> import pandas as pd
    >>>
    >>> # Quick detection
    >>> result = detect_patterns(df)
    >>> print(f"Found {len(result)} patterns")
    >>>
    >>> # Advanced usage
    >>> detector = ChartPatternDetector(min_confidence=0.7)
    >>> triangles = detector.detect_triangles(df, triangle_type='ascending')
    >>> h_and_s = detector.detect_head_and_shoulders(df)
"""

# Main detection classes and functions
from .chart_patterns import (
    ChartPatternDetector,
    detect_patterns
)

# Models
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
    TrendLine,
    PatternResult,
    PatternType
)

# Geometry utilities
from .geometry import (
    find_peaks,
    find_valleys,
    fit_trendline,
    create_trendline,
    find_support_resistance,
    calculate_angle,
    lines_intersect,
    calculate_pattern_confidence,
    is_price_near_line,
    count_trendline_touches,
    calculate_triangle_apex,
    smooth_prices
)

__all__ = [
    # Main API
    'ChartPatternDetector',
    'detect_patterns',

    # Pattern models
    'ChartPattern',
    'TrianglePattern',
    'HeadAndShouldersPattern',
    'DoubleToppingPattern',
    'TripleToppingPattern',
    'WedgePattern',
    'FlagPattern',
    'PennantPattern',
    'Point',
    'TrendLine',
    'PatternResult',
    'PatternType',

    # Geometry utilities
    'find_peaks',
    'find_valleys',
    'fit_trendline',
    'create_trendline',
    'find_support_resistance',
    'calculate_angle',
    'lines_intersect',
    'calculate_pattern_confidence',
    'is_price_near_line',
    'count_trendline_touches',
    'calculate_triangle_apex',
    'smooth_prices',
]

__version__ = '2.0.0'
