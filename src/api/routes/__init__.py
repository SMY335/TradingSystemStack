"""
API routes.
"""
from .data import router as data_router
from .indicators import router as indicators_router
from .candlesticks import router as candlesticks_router
from .vwap import router as vwap_router
from .zones import router as zones_router

__all__ = [
    'data_router',
    'indicators_router',
    'candlesticks_router',
    'vwap_router',
    'zones_router',
]
