"""
Market Scanner Module.

This module provides a DSL-based market scanner for filtering and ranking
securities based on technical and fundamental criteria.

Examples:
    >>> from src.scanner import ScanDefinition, run_scan, load_scan_from_json
    >>>
    >>> # Load scan from JSON
    >>> scan = load_scan_from_json('scans/rsi_oversold.json')
    >>> results = run_scan(scan, verbose=True)
    >>>
    >>> # Access results
    >>> for match in results.matched:
    ...     print(f"{match.symbol}: RSI={match.values.get('rsi', 'N/A')}")
"""

import json
from pathlib import Path
from typing import Union

from .dsl import (
    ScanDefinition,
    ScanResult,
    ScanResults,
    IndicatorConfig,
    Condition,
    Comparison,
    CrossCondition,
    PatternCondition,
    LogicalCondition
)
from .engine import ScanEngine, run_scan
from .operators import (
    compare,
    crosses_above,
    crosses_below,
    logical_and,
    logical_or,
    check_pattern,
    get_value
)


def load_scan_from_json(filepath: Union[str, Path]) -> ScanDefinition:
    """
    Load scan definition from JSON file.

    Args:
        filepath: Path to JSON scan definition

    Returns:
        ScanDefinition object

    Examples:
        >>> scan = load_scan_from_json('scans/momentum.json')
        >>> print(scan.name)
        'Momentum Breakout'

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If JSON is invalid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Scan file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return ScanDefinition(**data)


def save_scan_to_json(scan: ScanDefinition, filepath: Union[str, Path]) -> None:
    """
    Save scan definition to JSON file.

    Args:
        scan: ScanDefinition to save
        filepath: Output file path

    Examples:
        >>> scan = ScanDefinition(name='My Scan', universe=['AAPL'], ...)
        >>> save_scan_to_json(scan, 'scans/my_scan.json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(scan.model_dump(), f, indent=2, default=str)


def create_simple_scan(
    name: str,
    universe: list,
    indicator_name: str,
    operator: str,
    threshold: float,
    timeframe: str = '1d',
    lookback: int = 100
) -> ScanDefinition:
    """
    Create a simple single-indicator scan.

    Args:
        name: Scan name
        universe: List of symbols
        indicator_name: Indicator to use (e.g., 'rsi', 'macd')
        operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        threshold: Threshold value
        timeframe: Timeframe (default '1d')
        lookback: Lookback period (default 100)

    Returns:
        ScanDefinition

    Examples:
        >>> scan = create_simple_scan(
        ...     name='RSI Oversold',
        ...     universe=['AAPL', 'MSFT', 'GOOGL'],
        ...     indicator_name='rsi',
        ...     operator='<',
        ...     threshold=30
        ... )
    """
    return ScanDefinition(
        name=name,
        universe=universe,
        timeframe=timeframe,
        lookback=lookback,
        indicators=[
            IndicatorConfig(name=indicator_name)
        ],
        conditions=Comparison(
            type='comparison',
            left=indicator_name.lower(),
            operator=operator,
            right=threshold
        )
    )


def create_cross_scan(
    name: str,
    universe: list,
    fast_indicator: str,
    slow_indicator: str,
    direction: str = 'above',
    timeframe: str = '1d',
    lookback: int = 100
) -> ScanDefinition:
    """
    Create a crossover scan.

    Args:
        name: Scan name
        universe: List of symbols
        fast_indicator: Fast indicator/MA
        slow_indicator: Slow indicator/MA
        direction: 'above' or 'below'
        timeframe: Timeframe
        lookback: Lookback period

    Returns:
        ScanDefinition

    Examples:
        >>> scan = create_cross_scan(
        ...     name='Golden Cross',
        ...     universe=['SPY', 'QQQ'],
        ...     fast_indicator='sma_50',
        ...     slow_indicator='sma_200',
        ...     direction='above'
        ... )
    """
    # Parse indicator names to create configs
    indicators = []

    # Extract periods from indicator names (e.g., 'sma_50' -> period=50)
    for ind_name in [fast_indicator, slow_indicator]:
        parts = ind_name.split('_')
        indicator_type = parts[0]
        period = int(parts[1]) if len(parts) > 1 else 20

        indicators.append(
            IndicatorConfig(
                name=indicator_type.upper(),
                params={'period': period},
                alias=ind_name
            )
        )

    return ScanDefinition(
        name=name,
        universe=universe,
        timeframe=timeframe,
        lookback=lookback,
        indicators=indicators,
        conditions=CrossCondition(
            type='cross',
            series1=fast_indicator,
            series2=slow_indicator,
            direction=direction,
            lookback=5
        )
    )


__all__ = [
    # Core classes
    'ScanDefinition',
    'ScanResult',
    'ScanResults',
    'ScanEngine',

    # Condition types
    'IndicatorConfig',
    'Condition',
    'Comparison',
    'CrossCondition',
    'PatternCondition',
    'LogicalCondition',

    # Functions
    'run_scan',
    'load_scan_from_json',
    'save_scan_to_json',
    'create_simple_scan',
    'create_cross_scan',

    # Operators
    'compare',
    'crosses_above',
    'crosses_below',
    'logical_and',
    'logical_or',
    'check_pattern',
    'get_value',
]

__version__ = '2.0.0'
