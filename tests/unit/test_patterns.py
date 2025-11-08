"""
Unit tests for patterns module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.patterns import (
    ChartPatternDetector,
    detect_patterns,
    Point,
    TrendLine
)
from src.patterns.geometry import (
    find_peaks,
    find_valleys,
    fit_trendline,
    calculate_pattern_confidence
)


@pytest.fixture
def sample_ohlc():
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def test_find_peaks():
    """Test peak finding."""
    prices = np.array([1, 3, 2, 4, 3, 5, 2, 6, 4])
    peaks = find_peaks(prices, order=1)
    assert len(peaks) > 0


def test_find_valleys():
    """Test valley finding."""
    prices = np.array([3, 1, 2, 1, 3, 2, 4, 2, 5])
    valleys = find_valleys(prices, order=1)
    assert len(valleys) > 0


def test_fit_trendline():
    """Test trendline fitting."""
    indices = np.array([0, 1, 2, 3, 4])
    prices = np.array([100, 101, 102, 103, 104])
    slope, intercept, r2 = fit_trendline(indices, prices)

    assert slope > 0
    assert r2 > 0.9


def test_calculate_pattern_confidence():
    """Test confidence calculation."""
    confidence = calculate_pattern_confidence(0.95, 0.90, 0.85, 4)
    assert 0 <= confidence <= 1


def test_detector_initialization():
    """Test detector initialization."""
    detector = ChartPatternDetector(min_pattern_bars=20, min_confidence=0.5)
    assert detector.min_pattern_bars == 20
    assert detector.min_confidence == 0.5


def test_detect_triangles(sample_ohlc):
    """Test triangle detection."""
    detector = ChartPatternDetector(min_confidence=0.3)
    triangles = detector.detect_triangles(sample_ohlc, triangle_type='symmetrical')
    assert isinstance(triangles, list)


def test_detect_all_patterns(sample_ohlc):
    """Test detecting all patterns."""
    result = detect_patterns(sample_ohlc, min_confidence=0.3)
    assert hasattr(result, 'patterns')
    assert isinstance(result.patterns, list)


def test_point_creation():
    """Test Point dataclass."""
    point = Point(index=10, timestamp=datetime.now(), price=100.5)
    assert point.index == 10
    assert point.price == 100.5


def test_trendline_creation():
    """Test TrendLine dataclass."""
    points = [
        Point(0, datetime.now(), 100),
        Point(10, datetime.now(), 110)
    ]
    trendline = TrendLine(points=points, slope=1.0, intercept=100, r_squared=0.95)

    assert len(trendline.points) == 2
    assert trendline.slope == 1.0

    # Test price calculation
    price_at_5 = trendline.get_price_at_index(5)
    assert price_at_5 == 105.0
