"""
Sentiment aggregation from multiple sources.

This module combines sentiment signals from various sources (news, social media,
fear & greed index) into unified sentiment scores.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SentimentScore:
    """Aggregated sentiment score."""
    timestamp: datetime
    symbol: str
    composite_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    sources: Dict[str, float]  # Individual source scores
    signal: str  # 'bullish', 'bearish', 'neutral'

    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.composite_score > 0.2

    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.composite_score < -0.2

    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return not (self.is_bullish() or self.is_bearish())


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.

    Examples:
        >>> aggregator = SentimentAggregator()
        >>> score = aggregator.calculate_sentiment('AAPL', {
        ...     'fear_greed': 0.3,
        ...     'news': 0.5,
        ...     'social': 0.2
        ... })
        >>> print(f"Sentiment: {score.signal}")
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_sources: int = 1
    ):
        """
        Initialize aggregator.

        Args:
            weights: Weights for each source (must sum to 1.0)
            min_sources: Minimum number of sources required
        """
        self.weights = weights or {
            'fear_greed': 0.3,
            'news': 0.4,
            'social': 0.2,
            'technical': 0.1
        }
        self.min_sources = min_sources

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.weights = {k: v/total for k, v in self.weights.items()}

    def calculate_sentiment(
        self,
        symbol: str,
        source_scores: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> SentimentScore:
        """
        Calculate aggregated sentiment from multiple sources.

        Args:
            symbol: Stock/crypto symbol
            source_scores: Dictionary of source_name -> score (-1 to +1)
            timestamp: Timestamp for the score

        Returns:
            SentimentScore object

        Examples:
            >>> score = aggregator.calculate_sentiment('BTC', {
            ...     'fear_greed': 0.5,
            ...     'news': 0.3
            ... })
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check minimum sources
        if len(source_scores) < self.min_sources:
            return SentimentScore(
                timestamp=timestamp,
                symbol=symbol,
                composite_score=0.0,
                confidence=0.0,
                sources=source_scores,
                signal='neutral'
            )

        # Calculate weighted average
        composite = 0.0
        total_weight = 0.0

        for source, score in source_scores.items():
            if source in self.weights:
                weight = self.weights[source]
                composite += score * weight
                total_weight += weight

        # Normalize by actual weights used
        if total_weight > 0:
            composite /= total_weight

        # Calculate confidence based on agreement between sources
        if len(source_scores) > 1:
            scores = list(source_scores.values())
            std_dev = np.std(scores)
            # Lower std = higher confidence (sources agree)
            confidence = 1.0 - min(std_dev / 2.0, 1.0)
        else:
            confidence = 0.5  # Lower confidence with single source

        # Adjust confidence by number of sources
        source_factor = min(len(source_scores) / len(self.weights), 1.0)
        confidence *= source_factor

        # Determine signal
        if composite > 0.2:
            signal = 'bullish'
        elif composite < -0.2:
            signal = 'bearish'
        else:
            signal = 'neutral'

        return SentimentScore(
            timestamp=timestamp,
            symbol=symbol,
            composite_score=composite,
            confidence=confidence,
            sources=source_scores.copy(),
            signal=signal
        )

    def calculate_batch(
        self,
        symbols: List[str],
        source_scores_dict: Dict[str, Dict[str, float]]
    ) -> Dict[str, SentimentScore]:
        """
        Calculate sentiment for multiple symbols.

        Args:
            symbols: List of symbols
            source_scores_dict: Dict of symbol -> {source -> score}

        Returns:
            Dict of symbol -> SentimentScore

        Examples:
            >>> scores = aggregator.calculate_batch(
            ...     ['AAPL', 'MSFT'],
            ...     {
            ...         'AAPL': {'news': 0.5, 'social': 0.3},
            ...         'MSFT': {'news': 0.2, 'social': 0.1}
            ...     }
            ... )
        """
        results = {}

        for symbol in symbols:
            source_scores = source_scores_dict.get(symbol, {})
            results[symbol] = self.calculate_sentiment(symbol, source_scores)

        return results

    def to_dataframe(
        self,
        sentiment_scores: Dict[str, SentimentScore]
    ) -> pd.DataFrame:
        """
        Convert sentiment scores to DataFrame.

        Args:
            sentiment_scores: Dict of symbol -> SentimentScore

        Returns:
            DataFrame with sentiment data

        Examples:
            >>> df = aggregator.to_dataframe(scores)
            >>> print(df[['symbol', 'composite_score', 'signal']])
        """
        records = []

        for symbol, score in sentiment_scores.items():
            record = {
                'symbol': symbol,
                'timestamp': score.timestamp,
                'composite_score': score.composite_score,
                'confidence': score.confidence,
                'signal': score.signal,
                **{f'{k}_score': v for k, v in score.sources.items()}
            }
            records.append(record)

        return pd.DataFrame(records)


def calculate_sentiment_divergence(
    price_change: float,
    sentiment_change: float
) -> str:
    """
    Detect divergence between price and sentiment.

    Args:
        price_change: Price change percentage
        sentiment_change: Sentiment change (-1 to +1)

    Returns:
        Divergence type: 'bullish', 'bearish', or 'none'

    Examples:
        >>> div = calculate_sentiment_divergence(-5.0, 0.3)
        >>> print(div)  # 'bullish' - price down but sentiment up
    """
    # Normalize price change to -1 to +1 range
    price_norm = np.tanh(price_change / 10)  # Soft limit at ±10%

    # Check for divergence
    if price_norm < -0.2 and sentiment_change > 0.2:
        return 'bullish'  # Price falling but sentiment improving
    elif price_norm > 0.2 and sentiment_change < -0.2:
        return 'bearish'  # Price rising but sentiment deteriorating
    else:
        return 'none'
