"""
Pydantic request models for API.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class OHLCVRequest(BaseModel):
    """Request model for OHLCV data."""
    symbol: str = Field(..., description="Symbol (e.g., AAPL, BTC/USDT)")
    start: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    interval: str = Field('1d', description="Timeframe (1m, 5m, 1h, 1d, etc.)")
    source: str = Field('auto', description="Data source (auto, yfinance, ccxt)")


class IndicatorRequest(BaseModel):
    """Request model for indicator calculation."""
    symbol: str = Field(..., description="Symbol")
    indicator: str = Field(..., description="Indicator name (RSI, MACD, EMA, etc.)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")
    start: Optional[str] = Field(None, description="Start date")
    end: Optional[str] = Field(None, description="End date")
    interval: str = Field('1d', description="Timeframe")


class CandlestickRequest(BaseModel):
    """Request model for candlestick pattern detection."""
    symbol: str = Field(..., description="Symbol")
    patterns: Optional[List[str]] = Field(None, description="Patterns to detect (None = all)")
    start: Optional[str] = Field(None, description="Start date")
    end: Optional[str] = Field(None, description="End date")
    interval: str = Field('1d', description="Timeframe")


class VWAPRequest(BaseModel):
    """Request model for anchored VWAP."""
    symbol: str = Field(..., description="Symbol")
    anchor_type: str = Field('session', description="Anchor type (session, week, month, etc.)")
    include_bands: bool = Field(True, description="Include standard deviation bands")
    num_std: float = Field(1.0, description="Number of standard deviations for bands")
    start: Optional[str] = Field(None, description="Start date")
    end: Optional[str] = Field(None, description="End date")
    interval: str = Field('1d', description="Timeframe")


class ZonesRequest(BaseModel):
    """Request model for supply/demand zones."""
    symbol: str = Field(..., description="Symbol")
    consolidation_bars: int = Field(3, description="Minimum consolidation bars")
    impulse_threshold: float = Field(0.02, description="Minimum impulse move (0.02 = 2%)")
    max_zone_size: float = Field(0.03, description="Maximum zone size (0.03 = 3%)")
    start: Optional[str] = Field(None, description="Start date")
    end: Optional[str] = Field(None, description="End date")
    interval: str = Field('1d', description="Timeframe")
