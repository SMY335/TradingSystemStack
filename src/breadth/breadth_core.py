"""
Market breadth indicators.

This module calculates various market breadth indicators like percentage above SMA,
advance-decline line, and McClellan oscillator.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_percent_above_sma(
    prices_dict: Dict[str, pd.Series],
    sma_period: int = 200
) -> pd.Series:
    """
    Calculate percentage of stocks above their SMA.

    Args:
        prices_dict: Dictionary of symbol -> price Series
        sma_period: SMA period (default 200-day)

    Returns:
        Series with percentage above SMA (0-100)

    Examples:
        >>> prices = {'AAPL': df['close'], 'MSFT': df2['close']}
        >>> pct = calculate_percent_above_sma(prices, sma_period=200)
    """
    results = []

    for symbol, prices in prices_dict.items():
        sma = prices.rolling(window=sma_period).mean()
        above_sma = (prices > sma).astype(int)
        results.append(above_sma)

    # Calculate percentage
    df = pd.concat(results, axis=1)
    percent_above = (df.sum(axis=1) / len(prices_dict)) * 100

    return percent_above


def calculate_advance_decline_line(
    returns_dict: Dict[str, pd.Series]
) -> pd.Series:
    """
    Calculate advance-decline line.

    Args:
        returns_dict: Dictionary of symbol -> returns Series

    Returns:
        Cumulative advance-decline line

    Examples:
        >>> ad_line = calculate_advance_decline_line(returns)
    """
    advances = []
    declines = []

    for symbol, returns in returns_dict.items():
        advances.append((returns > 0).astype(int))
        declines.append((returns < 0).astype(int))

    df_adv = pd.concat(advances, axis=1)
    df_dec = pd.concat(declines, axis=1)

    net_advances = df_adv.sum(axis=1) - df_dec.sum(axis=1)
    ad_line = net_advances.cumsum()

    return ad_line


def calculate_mcclellan_oscillator(
    ad_line: pd.Series,
    fast_period: int = 19,
    slow_period: int = 39
) -> pd.Series:
    """
    Calculate McClellan Oscillator from A/D line.

    Args:
        ad_line: Advance-decline line
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        McClellan Oscillator values

    Examples:
        >>> ad = calculate_advance_decline_line(returns)
        >>> mcclellan = calculate_mcclellan_oscillator(ad)
    """
    fast_ema = ad_line.ewm(span=fast_period, adjust=False).mean()
    slow_ema = ad_line.ewm(span=slow_period, adjust=False).mean()

    oscillator = fast_ema - slow_ema

    return oscillator


def calculate_breadth_thrust(
    ad_line: pd.Series,
    window: int = 10,
    threshold: float = 0.615
) -> pd.Series:
    """
    Calculate breadth thrust indicator.

    Args:
        ad_line: Advance-decline line
        window: Rolling window
        threshold: Thrust threshold (default 61.5%)

    Returns:
        Boolean Series indicating thrust signals

    Examples:
        >>> thrust = calculate_breadth_thrust(ad_line)
    """
    # Calculate 10-day EMA of A/D ratio
    ad_ratio = ad_line / ad_line.rolling(window=window).sum()
    thrust = ad_ratio > threshold

    return thrust


class MarketBreadthAnalyzer:
    """
    Analyzer for market breadth indicators.

    Examples:
        >>> analyzer = MarketBreadthAnalyzer()
        >>> breadth = analyzer.analyze(universe_data)
    """

    def __init__(self, sma_period: int = 200):
        self.sma_period = sma_period

    def analyze(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Calculate all breadth indicators for a universe.

        Args:
            prices_dict: Dict of symbol -> DataFrame with prices
            price_column: Column name for prices

        Returns:
            DataFrame with breadth indicators

        Examples:
            >>> breadth_df = analyzer.analyze(universe_prices)
        """
        # Extract price series
        price_series = {
            symbol: df[price_column]
            for symbol, df in prices_dict.items()
        }

        # Calculate returns
        returns_series = {
            symbol: prices.pct_change()
            for symbol, prices in price_series.items()
        }

        # Calculate indicators
        pct_above_sma = calculate_percent_above_sma(price_series, self.sma_period)
        ad_line = calculate_advance_decline_line(returns_series)
        mcclellan = calculate_mcclellan_oscillator(ad_line)

        # Combine into DataFrame
        result = pd.DataFrame({
            'pct_above_sma': pct_above_sma,
            'ad_line': ad_line,
            'mcclellan': mcclellan
        })

        return result

    def get_breadth_signal(self, breadth_df: pd.DataFrame) -> str:
        """
        Get overall market breadth signal.

        Args:
            breadth_df: DataFrame from analyze()

        Returns:
            Signal: 'bullish', 'bearish', or 'neutral'

        Examples:
            >>> signal = analyzer.get_breadth_signal(breadth_df)
        """
        latest = breadth_df.iloc[-1]

        bullish_signals = 0
        bearish_signals = 0

        # Check percent above SMA
        if latest['pct_above_sma'] > 60:
            bullish_signals += 1
        elif latest['pct_above_sma'] < 40:
            bearish_signals += 1

        # Check A/D line trend
        if len(breadth_df) >= 20:
            ad_slope = breadth_df['ad_line'].iloc[-20:].diff().mean()
            if ad_slope > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

        # Check McClellan
        if latest['mcclellan'] > 50:
            bullish_signals += 1
        elif latest['mcclellan'] < -50:
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
