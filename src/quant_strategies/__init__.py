"""
Quantitative Trading Strategies Package

This package implements advanced quantitative trading strategies:
- Pairs Trading: Statistical arbitrage based on cointegration
- Mean Reversion: Statistical approaches to price mean reversion
"""

from src.quant_strategies.pairs_trading import PairsTradingStrategy

__all__ = [
    'PairsTradingStrategy',
]
