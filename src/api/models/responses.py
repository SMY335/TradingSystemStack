"""
Pydantic response models for API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class DataResponse(BaseModel):
    """Generic data response."""
    symbol: str
    data: List[Dict[str, Any]]
    rows: int
    columns: List[str]


class IndicatorResponse(BaseModel):
    """Indicator calculation response."""
    symbol: str
    indicator: str
    params: Dict[str, Any]
    data: List[Dict[str, Any]]
    rows: int


class PatternResponse(BaseModel):
    """Candlestick pattern response."""
    symbol: str
    patterns_detected: List[Dict[str, Any]]
    total_patterns: int


class VWAPResponse(BaseModel):
    """VWAP response."""
    symbol: str
    anchor_type: str
    data: List[Dict[str, Any]]
    rows: int


class ZonesResponse(BaseModel):
    """Supply/demand zones response."""
    symbol: str
    zones: List[Dict[str, Any]]
    total_zones: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
