"""
Sentiment Analysis Module.

This module provides sentiment analysis from multiple sources including
Fear & Greed Index, news sentiment, and social media sentiment.

Examples:
    >>> from src.sentiment import get_fear_greed_index, SentimentAggregator
    >>>
    >>> # Get Fear & Greed Index
    >>> fg = get_fear_greed_index()
    >>> print(f"Market sentiment: {fg.classification}")
    >>>
    >>> # Aggregate multiple sources
    >>> aggregator = SentimentAggregator()
    >>> score = aggregator.calculate_sentiment('AAPL', {
    ...     'fear_greed': 0.3,
    ...     'news': 0.5
    ... })
"""

# Fear & Greed Index
from .fear_greed import (
    FearGreedReading,
    get_fear_greed_index,
    get_fear_greed_history,
    classify_fear_greed,
    calculate_sentiment_momentum
)

# Sentiment aggregation
from .aggregator import (
    SentimentScore,
    SentimentAggregator,
    calculate_sentiment_divergence
)

__all__ = [
    # Fear & Greed
    'FearGreedReading',
    'get_fear_greed_index',
    'get_fear_greed_history',
    'classify_fear_greed',
    'calculate_sentiment_momentum',

    # Aggregation
    'SentimentScore',
    'SentimentAggregator',
    'calculate_sentiment_divergence',
]

__version__ = '2.0.0'

# Note: news.py, social.py, and nlp.py modules are planned for future implementation
# and will provide:
# - News sentiment analysis (NewsAPI, Finnhub)
# - Social media sentiment (Twitter, Reddit)
# - NLP processing (FinBERT, sentiment models)
