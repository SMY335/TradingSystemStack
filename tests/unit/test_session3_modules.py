"""
Quick smoke tests for Session 3 modules.
"""
import pytest
import pandas as pd
import numpy as np

# Trendlines
from src.trendlines import detect_support_resistance, TrendlineDetector

# Sentiment
from src.sentiment import get_fear_greed_index, SentimentAggregator

# Breadth
from src.breadth import MarketBreadthAnalyzer, calculate_percent_above_sma

# Relative Returns
from src.relativereturns import calculate_relative_strength, rank_by_rs

# Raindrop
from src.raindrop import is_plotly_available, calculate_volume_profile


@pytest.fixture
def sample_df():
    """Create sample DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)


class TestTrendlines:
    """Test trendlines module."""

    def test_detector_initialization(self):
        """Test TrendlineDetector initialization."""
        detector = TrendlineDetector(min_touches=2, min_strength=30)
        assert detector.min_touches == 2
        assert detector.min_strength == 30

    def test_detect_support_resistance(self, sample_df):
        """Test support/resistance detection."""
        trendlines = detect_support_resistance(sample_df, min_touches=2, min_strength=20)
        assert isinstance(trendlines, list)

    def test_trendline_detector(self, sample_df):
        """Test full detection workflow."""
        detector = TrendlineDetector(min_touches=2, min_strength=20)
        trendlines = detector.detect(sample_df, mode='both')
        assert isinstance(trendlines, list)


class TestSentiment:
    """Test sentiment module."""

    def test_fear_greed_index(self):
        """Test Fear & Greed Index fetching."""
        reading = get_fear_greed_index(use_mock=True)
        assert 0 <= reading.value <= 100
        assert reading.classification in [
            'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        ]

    def test_sentiment_aggregator(self):
        """Test sentiment aggregation."""
        aggregator = SentimentAggregator()
        score = aggregator.calculate_sentiment('AAPL', {
            'fear_greed': 0.3,
            'news': 0.5
        })

        assert -1 <= score.composite_score <= 1
        assert 0 <= score.confidence <= 1
        assert score.signal in ['bullish', 'bearish', 'neutral']

    def test_aggregator_batch(self):
        """Test batch sentiment calculation."""
        aggregator = SentimentAggregator()
        scores = aggregator.calculate_batch(
            ['AAPL', 'MSFT'],
            {
                'AAPL': {'news': 0.5},
                'MSFT': {'news': 0.3}
            }
        )

        assert 'AAPL' in scores
        assert 'MSFT' in scores


class TestBreadth:
    """Test breadth module."""

    def test_percent_above_sma(self):
        """Test percent above SMA calculation."""
        prices = {
            'A': pd.Series(np.random.randn(200).cumsum() + 100),
            'B': pd.Series(np.random.randn(200).cumsum() + 100)
        }

        pct = calculate_percent_above_sma(prices, sma_period=50)
        assert len(pct) == 200
        assert (pct >= 0).all() and (pct <= 100).all()

    def test_market_breadth_analyzer(self):
        """Test MarketBreadthAnalyzer."""
        analyzer = MarketBreadthAnalyzer(sma_period=50)
        assert analyzer.sma_period == 50


class TestRelativeReturns:
    """Test relative returns module."""

    def test_calculate_relative_strength(self):
        """Test RS calculation."""
        asset = pd.Series(np.random.randn(100).cumsum() + 100)
        benchmark = pd.Series(np.random.randn(100).cumsum() + 100)

        rs = calculate_relative_strength(asset, benchmark)
        assert len(rs) == 100
        assert (rs > 0).all()  # Ratios should be positive

    def test_rank_by_rs(self):
        """Test RS ranking."""
        prices = {
            'A': pd.Series(np.random.randn(100).cumsum() + 100),
            'B': pd.Series(np.random.randn(100).cumsum() + 100),
            'C': pd.Series(np.random.randn(100).cumsum() + 100)
        }
        benchmark = pd.Series(np.random.randn(100).cumsum() + 100)

        rankings = rank_by_rs(prices, benchmark)
        assert len(rankings) == 3
        assert 'rs_rating' in rankings.columns
        assert 'rank' in rankings.columns


class TestRaindrop:
    """Test raindrop module."""

    def test_plotly_availability(self):
        """Test plotly availability check."""
        available = is_plotly_available()
        assert isinstance(available, bool)

    def test_calculate_volume_profile(self, sample_df):
        """Test volume profile calculation."""
        prices, vol_up, vol_down = calculate_volume_profile(sample_df, price_bins=20)

        assert len(prices) == 19  # bins - 1
        assert len(vol_up) == 19
        assert len(vol_down) == 19
        assert (vol_up >= 0).all()
        assert (vol_down >= 0).all()
