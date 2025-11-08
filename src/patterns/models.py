"""
Pattern models and data structures.

This module defines dataclasses for various chart patterns detected in price data.
"""
from dataclasses import dataclass
from typing import Optional, List, Literal
from datetime import datetime


PatternType = Literal[
    'triangle_ascending',
    'triangle_descending',
    'triangle_symmetrical',
    'head_and_shoulders',
    'inverse_head_and_shoulders',
    'double_top',
    'double_bottom',
    'triple_top',
    'triple_bottom',
    'wedge_rising',
    'wedge_falling',
    'flag_bullish',
    'flag_bearish',
    'pennant'
]


@dataclass
class Point:
    """Price point with index and value."""
    index: int
    timestamp: datetime
    price: float

    def __repr__(self) -> str:
        return f"Point(idx={self.index}, price={self.price:.2f})"


@dataclass
class TrendLine:
    """Trend line defined by two or more points."""
    points: List[Point]
    slope: float
    intercept: float
    r_squared: float  # Goodness of fit

    def get_price_at_index(self, index: int) -> float:
        """Calculate price at given index using line equation."""
        return self.slope * index + self.intercept

    def __repr__(self) -> str:
        return f"TrendLine(points={len(self.points)}, slope={self.slope:.4f}, R²={self.r_squared:.3f})"


@dataclass
class ChartPattern:
    """Base class for all chart patterns."""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    confidence: float  # 0.0 to 1.0
    price_range: tuple  # (min, max)
    breakout_target: Optional[float] = None
    breakout_direction: Optional[Literal['up', 'down']] = None

    def duration(self) -> int:
        """Number of bars in pattern."""
        return self.end_idx - self.start_idx + 1

    def height(self) -> float:
        """Price range of pattern."""
        return self.price_range[1] - self.price_range[0]

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(type={self.pattern_type}, "
                f"duration={self.duration()}, confidence={self.confidence:.2f})")


@dataclass
class TrianglePattern:
    """Triangle pattern (ascending, descending, or symmetrical)."""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    confidence: float
    price_range: tuple
    upper_trendline: TrendLine
    lower_trendline: TrendLine
    apex_index: int  # Where trendlines converge
    width: int  # Pattern width in bars
    breakout_target: Optional[float] = None
    breakout_direction: Optional[Literal['up', 'down']] = None

    def duration(self) -> int:
        """Number of bars in pattern."""
        return self.end_idx - self.start_idx + 1

    def height(self) -> float:
        """Price range of pattern."""
        return self.price_range[1] - self.price_range[0]

    def is_converging(self) -> bool:
        """Check if trendlines are actually converging."""
        return abs(self.upper_trendline.slope) > 0 or abs(self.lower_trendline.slope) > 0

    def __repr__(self) -> str:
        return (f"TrianglePattern(type={self.pattern_type}, "
                f"duration={self.duration()}, confidence={self.confidence:.2f})")


@dataclass
class HeadAndShouldersPattern:
    """Head and shoulders or inverse head and shoulders pattern."""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    confidence: float
    price_range: tuple
    left_shoulder: Point
    head: Point
    right_shoulder: Point
    neckline: TrendLine
    breakout_target: Optional[float] = None
    breakout_direction: Optional[Literal['up', 'down']] = None
    neckline_breakout: Optional[Point] = None

    def duration(self) -> int:
        """Number of bars in pattern."""
        return self.end_idx - self.start_idx + 1

    def height(self) -> float:
        """Price range of pattern."""
        return self.price_range[1] - self.price_range[0]

    def symmetry_score(self) -> float:
        """Calculate symmetry between shoulders (0.0 to 1.0)."""
        left_to_head = abs(self.head.index - self.left_shoulder.index)
        head_to_right = abs(self.right_shoulder.index - self.head.index)

        if left_to_head == 0 or head_to_right == 0:
            return 0.0

        ratio = min(left_to_head, head_to_right) / max(left_to_head, head_to_right)
        return ratio


@dataclass
class DoubleToppingPattern:
    """Double top or double bottom pattern."""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    confidence: float
    price_range: tuple
    first_peak: Point
    valley: Point
    second_peak: Point
    support_resistance_level: float
    peak_symmetry: float  # How similar the two peaks are (0.0 to 1.0)
    breakout_target: Optional[float] = None
    breakout_direction: Optional[Literal['up', 'down']] = None

    def duration(self) -> int:
        """Number of bars in pattern."""
        return self.end_idx - self.start_idx + 1

    def height(self) -> float:
        """Price range of pattern."""
        return self.price_range[1] - self.price_range[0]

    def is_valid_double(self, tolerance: float = 0.03) -> bool:
        """Check if peaks are within tolerance of each other."""
        price_diff = abs(self.first_peak.price - self.second_peak.price)
        avg_price = (self.first_peak.price + self.second_peak.price) / 2
        return (price_diff / avg_price) <= tolerance


# Note: Additional pattern types below are defined but not yet implemented in detector
# They can be implemented in future versions

@dataclass
class TripleToppingPattern:
    """Triple top or triple bottom pattern (placeholder for future implementation)."""
    pattern_type: PatternType
    first_peak: Point
    second_peak: Point
    third_peak: Point
    support_resistance_level: float

    def is_valid_triple(self, tolerance: float = 0.03) -> bool:
        """Check if all three peaks are within tolerance."""
        prices = [self.first_peak.price, self.second_peak.price, self.third_peak.price]
        avg_price = sum(prices) / 3
        max_diff = max(abs(p - avg_price) for p in prices)
        return (max_diff / avg_price) <= tolerance


@dataclass
class WedgePattern:
    """Rising or falling wedge pattern (placeholder for future implementation)."""
    pattern_type: PatternType
    upper_trendline: TrendLine
    lower_trendline: TrendLine
    apex_index: int

    def convergence_angle(self) -> float:
        """Angle between upper and lower trendlines (in degrees)."""
        import math
        slope_diff = abs(self.upper_trendline.slope - self.lower_trendline.slope)
        return math.degrees(math.atan(slope_diff))


@dataclass
class FlagPattern:
    """Flag pattern (placeholder for future implementation)."""
    pattern_type: PatternType
    pole_start: Point
    pole_end: Point
    flag_start: Point
    flag_end: Point
    pole_height: float

    def pole_strength(self) -> float:
        """Calculate strength of the pole move."""
        return abs(self.pole_height) / self.pole_start.price


@dataclass
class PennantPattern:
    """Pennant pattern (placeholder for future implementation)."""
    pattern_type: PatternType
    pole_start: Point
    pole_end: Point
    pennant_upper: TrendLine
    pennant_lower: TrendLine
    pole_height: float


@dataclass
class PatternResult:
    """Result of pattern detection containing multiple patterns."""
    patterns: List[ChartPattern]
    detection_time: datetime
    symbol: str
    timeframe: str

    def filter_by_confidence(self, min_confidence: float = 0.5) -> List[ChartPattern]:
        """Filter patterns by minimum confidence."""
        return [p for p in self.patterns if p.confidence >= min_confidence]

    def filter_by_type(self, pattern_type: PatternType) -> List[ChartPattern]:
        """Filter patterns by type."""
        return [p for p in self.patterns if p.pattern_type == pattern_type]

    def sort_by_confidence(self, descending: bool = True) -> List[ChartPattern]:
        """Sort patterns by confidence."""
        return sorted(self.patterns, key=lambda p: p.confidence, reverse=descending)

    def __len__(self) -> int:
        return len(self.patterns)

    def __repr__(self) -> str:
        return f"PatternResult(symbol={self.symbol}, patterns={len(self.patterns)})"
