"""
Fear & Greed Index fetching and analysis.

This module fetches market sentiment indicators like the CNN Fear & Greed Index
and provides historical data access.
"""
import pandas as pd
import numpy as np
from typing import Optional, Literal
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class FearGreedReading:
    """Single Fear & Greed Index reading."""
    date: datetime
    value: int  # 0-100
    classification: str  # 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
    previous_close: Optional[int] = None
    one_week_ago: Optional[int] = None
    one_month_ago: Optional[int] = None
    one_year_ago: Optional[int] = None

    def is_fearful(self) -> bool:
        """Check if market is in fear mode."""
        return self.value < 45

    def is_greedy(self) -> bool:
        """Check if market is in greed mode."""
        return self.value > 55

    def sentiment_score(self) -> float:
        """Normalize to -1 (extreme fear) to +1 (extreme greed)."""
        return (self.value - 50) / 50


def classify_fear_greed(value: int) -> str:
    """
    Classify Fear & Greed value into categories.

    Args:
        value: Fear & Greed Index value (0-100)

    Returns:
        Classification string

    Examples:
        >>> classify_fear_greed(25)
        'Fear'
        >>> classify_fear_greed(75)
        'Greed'
    """
    if value >= 75:
        return 'Extreme Greed'
    elif value >= 55:
        return 'Greed'
    elif value >= 45:
        return 'Neutral'
    elif value >= 25:
        return 'Fear'
    else:
        return 'Extreme Fear'


def get_fear_greed_index(
    use_mock: bool = True
) -> FearGreedReading:
    """
    Fetch current Fear & Greed Index.

    Args:
        use_mock: Use mock data (True) or attempt real API call (False)

    Returns:
        FearGreedReading object

    Examples:
        >>> reading = get_fear_greed_index()
        >>> print(f"Current sentiment: {reading.classification}")

    Note:
        Real API implementation would require CNN Fear & Greed API access.
        Currently uses mock data for development.
    """
    if use_mock:
        # Generate realistic mock data
        value = int(50 + np.random.randn() * 15)
        value = np.clip(value, 0, 100)

        return FearGreedReading(
            date=datetime.now(),
            value=value,
            classification=classify_fear_greed(value),
            previous_close=value + np.random.randint(-5, 6),
            one_week_ago=value + np.random.randint(-10, 11),
            one_month_ago=value + np.random.randint(-15, 16),
            one_year_ago=value + np.random.randint(-25, 26)
        )
    else:
        # Real implementation would fetch from API
        # Example: https://api.alternative.me/fng/ (for crypto)
        raise NotImplementedError("Real API integration not yet implemented")


def get_fear_greed_history(
    days: int = 30,
    use_mock: bool = True
) -> pd.DataFrame:
    """
    Fetch historical Fear & Greed Index data.

    Args:
        days: Number of days of history
        use_mock: Use mock data or real API

    Returns:
        DataFrame with columns: date, value, classification

    Examples:
        >>> history = get_fear_greed_history(days=7)
        >>> print(history.tail())
    """
    if use_mock:
        # Generate synthetic history with realistic trends
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate with some autocorrelation (sentiment persists)
        values = []
        current = 50 + np.random.randn() * 10

        for _ in range(days):
            # Random walk with mean reversion
            current = current * 0.9 + np.random.randn() * 5 + 50 * 0.1
            current = np.clip(current, 0, 100)
            values.append(int(current))

        df = pd.DataFrame({
            'date': dates,
            'value': values
        })

        df['classification'] = df['value'].apply(classify_fear_greed)

        return df
    else:
        raise NotImplementedError("Real API integration not yet implemented")


def calculate_sentiment_momentum(
    history: pd.DataFrame,
    window: int = 7
) -> float:
    """
    Calculate momentum in sentiment (rate of change).

    Args:
        history: DataFrame from get_fear_greed_history()
        window: Lookback window in days

    Returns:
        Momentum score (-100 to +100)

    Examples:
        >>> history = get_fear_greed_history(30)
        >>> momentum = calculate_sentiment_momentum(history, window=7)
    """
    if len(history) < window:
        return 0.0

    recent = history['value'].iloc[-window:].mean()
    older = history['value'].iloc[-(2*window):-window].mean()

    return recent - older
