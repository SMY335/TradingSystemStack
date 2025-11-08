"""
REST API for TradingSystemStack.

Run with: uvicorn src.api.main:app --reload
Access docs at: http://localhost:8000/docs
"""

from .main import app

__all__ = ['app']
