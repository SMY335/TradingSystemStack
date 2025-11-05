"""
Market Making Strategies Package

This package implements market making strategies:
- Simple Market Maker: Basic bid-ask spread with inventory management
"""

from src.market_making.simple_mm import SimpleMarketMaker

__all__ = [
    'SimpleMarketMaker',
]
