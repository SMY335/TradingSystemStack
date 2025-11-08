"""
Market Breadth Module.

This module provides market breadth analysis including percentage above SMA,
advance-decline line, and McClellan oscillator.

Examples:
    >>> from src.breadth import MarketBreadthAnalyzer, calculate_percent_above_sma
    >>>
    >>> analyzer = MarketBreadthAnalyzer()
    >>> breadth_df = analyzer.analyze(universe_prices)
    >>> signal = analyzer.get_breadth_signal(breadth_df)
"""

from .breadth_core import (
    MarketBreadthAnalyzer,
    calculate_percent_above_sma,
    calculate_advance_decline_line,
    calculate_mcclellan_oscillator,
    calculate_breadth_thrust
)

__all__ = [
    'MarketBreadthAnalyzer',
    'calculate_percent_above_sma',
    'calculate_advance_decline_line',
    'calculate_mcclellan_oscillator',
    'calculate_breadth_thrust',
]

__version__ = '2.0.0'
