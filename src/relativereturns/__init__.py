"""
Relative Returns and Strength Module.

This module provides relative strength analysis, ranking, and comparison
between assets and benchmarks.

Examples:
    >>> from src.relativereturns import calculate_relative_strength, rank_by_rs
    >>>
    >>> # Calculate RS vs benchmark
    >>> rs = calculate_relative_strength(asset_prices, benchmark_prices)
    >>>
    >>> # Rank universe by RS
    >>> rankings = rank_by_rs(prices_dict, benchmark)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def calculate_relative_strength(
    asset: pd.Series,
    benchmark: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate relative strength ratio (asset / benchmark).

    Args:
        asset: Asset price series
        benchmark: Benchmark price series
        window: Rolling window for normalization

    Returns:
        Relative strength ratio

    Examples:
        >>> rs = calculate_relative_strength(aapl_prices, spy_prices)
    """
    # Normalize both to starting value
    asset_norm = asset / asset.iloc[0] * 100
    benchmark_norm = benchmark / benchmark.iloc[0] * 100

    # Calculate ratio
    rs_ratio = asset_norm / benchmark_norm

    return rs_ratio


def calculate_rs_rating(
    asset: pd.Series,
    benchmark: pd.Series,
    lookback_periods: List[int] = [63, 126, 189, 252]
) -> float:
    """
    Calculate RS rating (0-100) based on multiple timeframes.

    Args:
        asset: Asset price series
        benchmark: Benchmark price series
        lookback_periods: List of lookback periods in days

    Returns:
        RS rating (0-100)

    Examples:
        >>> rating = calculate_rs_rating(stock_prices, index_prices)
    """
    if len(asset) < max(lookback_periods):
        return 50.0  # Neutral if insufficient data

    scores = []

    for period in lookback_periods:
        if len(asset) >= period:
            asset_return = (asset.iloc[-1] / asset.iloc[-period] - 1) * 100
            benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[-period] - 1) * 100

            relative_return = asset_return - benchmark_return
            scores.append(relative_return)

    # Average relative performance across timeframes
    avg_score = np.mean(scores)

    # Convert to 0-100 scale (assuming ±50% relative performance is extreme)
    rating = 50 + (avg_score / 50) * 50
    rating = np.clip(rating, 0, 100)

    return rating


def rank_by_rs(
    prices_dict: Dict[str, pd.Series],
    benchmark: pd.Series,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Rank assets by relative strength.

    Args:
        prices_dict: Dict of symbol -> price Series
        benchmark: Benchmark price series
        top_n: Return only top N (None for all)

    Returns:
        DataFrame with rankings

    Examples:
        >>> rankings = rank_by_rs(universe_prices, spy_prices, top_n=10)
    """
    results = []

    for symbol, prices in prices_dict.items():
        if len(prices) >= 63:  # Minimum data requirement
            rating = calculate_rs_rating(prices, benchmark)
            latest_rs = calculate_relative_strength(prices, benchmark).iloc[-1]

            results.append({
                'symbol': symbol,
                'rs_rating': rating,
                'rs_ratio': latest_rs,
                'rank': 0  # Will be filled
            })

    # Create DataFrame and rank
    df = pd.DataFrame(results)

    if len(df) > 0:
        df = df.sort_values('rs_rating', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        if top_n is not None:
            df = df.head(top_n)

    return df


def calculate_rs_matrix(
    prices_dict: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Calculate relative strength matrix (all vs all).

    Args:
        prices_dict: Dict of symbol -> price Series

    Returns:
        DataFrame with RS ratios (rows vs columns)

    Examples:
        >>> matrix = calculate_rs_matrix(sector_etfs)
    """
    symbols = list(prices_dict.keys())
    matrix = pd.DataFrame(index=symbols, columns=symbols, dtype=float)

    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 == sym2:
                matrix.loc[sym1, sym2] = 1.0
            else:
                rs = calculate_relative_strength(
                    prices_dict[sym1],
                    prices_dict[sym2]
                )
                matrix.loc[sym1, sym2] = rs.iloc[-1]

    return matrix


def detect_rs_momentum_change(
    rs_series: pd.Series,
    window: int = 20
) -> Tuple[bool, str]:
    """
    Detect momentum change in relative strength.

    Args:
        rs_series: Relative strength time series
        window: Window for momentum calculation

    Returns:
        Tuple of (change_detected, direction)

    Examples:
        >>> changed, direction = detect_rs_momentum_change(rs)
    """
    if len(rs_series) < window * 2:
        return False, 'neutral'

    # Calculate recent vs older momentum
    recent_momentum = rs_series.iloc[-window:].diff().mean()
    older_momentum = rs_series.iloc[-2*window:-window].diff().mean()

    # Check for significant change
    momentum_change = recent_momentum - older_momentum

    if abs(momentum_change) > 0.1:  # Threshold for significance
        direction = 'improving' if momentum_change > 0 else 'deteriorating'
        return True, direction
    else:
        return False, 'neutral'


class RelativeStrengthAnalyzer:
    """
    Analyzer for relative strength across universe.

    Examples:
        >>> analyzer = RelativeStrengthAnalyzer(benchmark=spy_prices)
        >>> rankings = analyzer.rank_universe(universe_prices)
        >>> leaders = analyzer.get_leaders(10)
    """

    def __init__(self, benchmark: pd.Series):
        self.benchmark = benchmark
        self._last_rankings = None

    def rank_universe(
        self,
        prices_dict: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Rank entire universe by RS."""
        self._last_rankings = rank_by_rs(prices_dict, self.benchmark)
        return self._last_rankings

    def get_leaders(self, top_n: int = 10) -> pd.DataFrame:
        """Get top RS leaders."""
        if self._last_rankings is None:
            return pd.DataFrame()
        return self._last_rankings.head(top_n)

    def get_laggards(self, bottom_n: int = 10) -> pd.DataFrame:
        """Get bottom RS laggards."""
        if self._last_rankings is None:
            return pd.DataFrame()
        return self._last_rankings.tail(bottom_n)


__all__ = [
    'calculate_relative_strength',
    'calculate_rs_rating',
    'rank_by_rs',
    'calculate_rs_matrix',
    'detect_rs_momentum_change',
    'RelativeStrengthAnalyzer',
]

__version__ = '2.0.0'
