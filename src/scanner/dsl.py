"""
Scanner DSL (Domain Specific Language) definitions.

This module defines the Pydantic models for the scanner JSON DSL,
allowing users to define market scans declaratively.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Union, Any
from datetime import datetime


class IndicatorConfig(BaseModel):
    """Configuration for an indicator."""
    name: str = Field(..., description="Indicator name (RSI, MACD, EMA, etc.)")
    params: dict = Field(default_factory=dict, description="Indicator parameters")
    alias: Optional[str] = Field(None, description="Alias for referencing in conditions")


class Comparison(BaseModel):
    """Comparison condition."""
    type: Literal['comparison'] = 'comparison'
    left: str = Field(..., description="Left operand (column name or alias)")
    operator: Literal['>', '<', '>=', '<=', '==', '!='] = Field(..., description="Comparison operator")
    right: Union[str, float, int] = Field(..., description="Right operand (column, value, or alias)")


class CrossCondition(BaseModel):
    """Crossover/crossunder condition."""
    type: Literal['cross'] = 'cross'
    series1: str = Field(..., description="First series")
    series2: str = Field(..., description="Second series")
    direction: Literal['above', 'below'] = Field(..., description="Cross direction")
    lookback: int = Field(5, description="Bars to look back for cross")


class PatternCondition(BaseModel):
    """Pattern detection condition."""
    type: Literal['pattern'] = 'pattern'
    pattern_type: str = Field(..., description="Pattern type (doji, hammer, etc.)")
    timeframe: str = Field('1d', description="Timeframe for pattern")


class LogicalCondition(BaseModel):
    """Logical combination of conditions (AND/OR)."""
    type: Literal['and', 'or'] = Field(..., description="Logical operator")
    conditions: List['Condition'] = Field(..., description="List of conditions to combine")


# Union type for all conditions
Condition = Union[Comparison, CrossCondition, PatternCondition, LogicalCondition]

# Update forward references
LogicalCondition.model_rebuild()


class ScanDefinition(BaseModel):
    """Complete scan definition."""
    name: str = Field(..., description="Scan name")
    description: Optional[str] = Field(None, description="Scan description")
    universe: List[str] = Field(..., description="List of symbols to scan")
    timeframe: str = Field('1d', description="Timeframe (1m, 5m, 1h, 1d, etc.)")
    lookback: int = Field(100, description="Number of bars to load")

    # Indicators to calculate
    indicators: List[IndicatorConfig] = Field(default_factory=list, description="Indicators to calculate")

    # Conditions to match
    conditions: Condition = Field(..., description="Condition tree")

    # Output configuration
    max_results: Optional[int] = Field(None, description="Maximum results to return")
    sort_by: Optional[str] = Field(None, description="Column to sort results by")
    sort_ascending: bool = Field(False, description="Sort direction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "RSI Oversold",
                "description": "Find stocks with RSI below 30",
                "universe": ["AAPL", "MSFT", "GOOGL"],
                "timeframe": "1d",
                "lookback": 100,
                "indicators": [
                    {
                        "name": "RSI",
                        "params": {"length": 14},
                        "alias": "rsi"
                    }
                ],
                "conditions": {
                    "type": "comparison",
                    "left": "rsi",
                    "operator": "<",
                    "right": 30
                },
                "max_results": 10,
                "sort_by": "rsi",
                "sort_ascending": True
            }
        }
    )


class ScanResult(BaseModel):
    """Result from a scan."""
    symbol: str
    matched: bool
    timestamp: datetime
    values: dict = Field(default_factory=dict, description="Relevant values at match time")


class ScanResults(BaseModel):
    """Results from scanning multiple symbols."""
    scan_name: str
    total_scanned: int
    matched: List[ScanResult]
    execution_time: float  # seconds
    timestamp: datetime


def load_scan_from_json(json_path: str) -> ScanDefinition:
    """
    Load scan definition from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        ScanDefinition object

    Examples:
        >>> scan = load_scan_from_json('scans/rsi_oversold.json')
    """
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    return ScanDefinition(**data)


def save_scan_to_json(scan: ScanDefinition, json_path: str):
    """
    Save scan definition to JSON file.

    Args:
        scan: ScanDefinition object
        json_path: Path to save JSON

    Examples:
        >>> save_scan_to_json(scan, 'scans/my_scan.json')
    """
    import json

    with open(json_path, 'w') as f:
        json.dump(scan.model_dump(), f, indent=2, default=str)
