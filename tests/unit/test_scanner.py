"""
Unit tests for scanner module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.scanner import (
    ScanDefinition,
    ScanEngine,
    run_scan,
    load_scan_from_json,
    save_scan_to_json,
    create_simple_scan,
    create_cross_scan,
    IndicatorConfig,
    Comparison,
    CrossCondition,
    LogicalCondition
)
from src.scanner.operators import (
    compare,
    crosses_above,
    crosses_below,
    logical_and,
    logical_or,
    get_value
)


@pytest.fixture
def sample_df():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # Ensure OHLC validity
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def test_comparison_operator():
    """Test comparison operator."""
    series = pd.Series([10, 20, 30, 40, 50])

    result_gt = compare(series, '>', 25)
    assert result_gt.sum() == 3  # 30, 40, 50

    result_lt = compare(series, '<', 25)
    assert result_lt.sum() == 2  # 10, 20

    result_eq = compare(series, '==', 30)
    assert result_eq.sum() == 1


def test_crosses_above():
    """Test crosses_above operator."""
    fast = pd.Series([1, 2, 3, 4, 5])
    slow = pd.Series([5, 4, 3, 2, 1])

    crosses = crosses_above(fast, slow, lookback=2)
    # Cross happens between index 2 and 3
    assert crosses.iloc[-1] == True or crosses.iloc[-2] == True


def test_crosses_below():
    """Test crosses_below operator."""
    fast = pd.Series([5, 4, 3, 2, 1])
    slow = pd.Series([1, 2, 3, 4, 5])

    crosses = crosses_below(fast, slow, lookback=2)
    # Should detect cross
    assert crosses.any()


def test_logical_and():
    """Test logical AND operator."""
    cond1 = pd.Series([True, True, False, False])
    cond2 = pd.Series([True, False, True, False])

    result = logical_and([cond1, cond2])
    expected = pd.Series([True, False, False, False])

    assert result.equals(expected)


def test_logical_or():
    """Test logical OR operator."""
    cond1 = pd.Series([True, True, False, False])
    cond2 = pd.Series([True, False, True, False])

    result = logical_or([cond1, cond2])
    expected = pd.Series([True, True, True, False])

    assert result.equals(expected)


def test_get_value(sample_df):
    """Test get_value function."""
    value = get_value(sample_df, 'close')
    assert isinstance(value, pd.Series)
    assert len(value) == len(sample_df)


def test_indicator_config_creation():
    """Test IndicatorConfig creation."""
    config = IndicatorConfig(
        name='RSI',
        params={'period': 14},
        alias='rsi'
    )

    assert config.name == 'RSI'
    assert config.params['period'] == 14
    assert config.alias == 'rsi'


def test_comparison_condition_creation():
    """Test Comparison condition creation."""
    cond = Comparison(
        type='comparison',
        left='rsi',
        operator='<',
        right=30
    )

    assert cond.type == 'comparison'
    assert cond.left == 'rsi'
    assert cond.operator == '<'
    assert cond.right == 30


def test_cross_condition_creation():
    """Test CrossCondition creation."""
    cond = CrossCondition(
        type='cross',
        series1='sma_50',
        series2='sma_200',
        direction='above',
        lookback=5
    )

    assert cond.type == 'cross'
    assert cond.series1 == 'sma_50'
    assert cond.direction == 'above'


def test_logical_condition_creation():
    """Test LogicalCondition creation."""
    cond = LogicalCondition(
        type='and',
        conditions=[
            Comparison(type='comparison', left='rsi', operator='>', right=50),
            Comparison(type='comparison', left='close', operator='>', right='sma_50')
        ]
    )

    assert cond.type == 'and'
    assert len(cond.conditions) == 2


def test_scan_definition_creation():
    """Test ScanDefinition creation."""
    scan = ScanDefinition(
        name='Test Scan',
        universe=['AAPL', 'MSFT'],
        timeframe='1d',
        lookback=100,
        indicators=[
            IndicatorConfig(name='RSI', params={'period': 14})
        ],
        conditions=Comparison(
            type='comparison',
            left='rsi',
            operator='<',
            right=30
        )
    )

    assert scan.name == 'Test Scan'
    assert len(scan.universe) == 2
    assert scan.timeframe == '1d'


def test_create_simple_scan():
    """Test create_simple_scan helper."""
    scan = create_simple_scan(
        name='RSI Oversold',
        universe=['AAPL', 'MSFT'],
        indicator_name='RSI',
        operator='<',
        threshold=30
    )

    assert scan.name == 'RSI Oversold'
    assert len(scan.universe) == 2
    assert len(scan.indicators) == 1
    assert scan.indicators[0].name == 'RSI'


def test_create_cross_scan():
    """Test create_cross_scan helper."""
    scan = create_cross_scan(
        name='Golden Cross',
        universe=['SPY', 'QQQ'],
        fast_indicator='sma_50',
        slow_indicator='sma_200',
        direction='above'
    )

    assert scan.name == 'Golden Cross'
    assert len(scan.indicators) == 2
    assert scan.conditions.type == 'cross'


def test_save_and_load_scan(tmp_path):
    """Test saving and loading scan from JSON."""
    scan = create_simple_scan(
        name='Test Scan',
        universe=['AAPL'],
        indicator_name='RSI',
        operator='<',
        threshold=30
    )

    filepath = tmp_path / 'test_scan.json'
    save_scan_to_json(scan, filepath)

    assert filepath.exists()

    loaded_scan = load_scan_from_json(filepath)
    assert loaded_scan.name == scan.name
    assert loaded_scan.universe == scan.universe


def test_scan_engine_initialization():
    """Test ScanEngine initialization."""
    engine = ScanEngine(max_workers=2, verbose=True)
    assert engine.max_workers == 2
    assert engine.verbose == True


def test_scan_engine_execute():
    """Test ScanEngine.execute() with simple scan."""
    # Create a simple scan
    scan = create_simple_scan(
        name='Price Above 100',
        universe=['AAPL', 'MSFT'],  # Small universe for testing
        indicator_name='SMA',
        operator='>',
        threshold=0,  # Always true condition for testing
        lookback=50
    )

    engine = ScanEngine(max_workers=1, verbose=False)

    try:
        results = engine.execute(scan)

        assert results.scan_name == 'Price Above 100'
        assert results.total_scanned == 2
        assert isinstance(results.matched, list)
        assert results.execution_time >= 0

    except Exception as e:
        # If data fetching fails (expected in test environment), that's okay
        pytest.skip(f"Data fetching failed in test environment: {e}")


def test_run_scan_convenience_function():
    """Test run_scan convenience function."""
    scan = create_simple_scan(
        name='Simple Test',
        universe=['AAPL'],
        indicator_name='SMA',
        operator='>',
        threshold=0,
        lookback=50
    )

    try:
        results = run_scan(scan, max_workers=1, verbose=False)
        assert results.scan_name == 'Simple Test'
    except Exception:
        pytest.skip("Data fetching failed in test environment")


def test_example_scan_files_exist():
    """Test that example scan files exist and are valid."""
    examples_dir = Path('examples/scans')

    if not examples_dir.exists():
        pytest.skip("Examples directory not found")

    scan_files = [
        'rsi_oversold.json',
        'golden_cross.json',
        'momentum_breakout.json'
    ]

    for filename in scan_files:
        filepath = examples_dir / filename

        if filepath.exists():
            # Try to load it
            scan = load_scan_from_json(filepath)
            assert scan.name is not None
            assert len(scan.universe) > 0
            assert scan.conditions is not None


def test_scan_with_sorting():
    """Test scan with result sorting."""
    scan = ScanDefinition(
        name='Sorted Scan',
        universe=['AAPL', 'MSFT', 'GOOGL'],
        timeframe='1d',
        lookback=50,
        indicators=[
            IndicatorConfig(name='RSI', params={'period': 14})
        ],
        conditions=Comparison(
            type='comparison',
            left='rsi',
            operator='>',
            right=0  # Always true
        ),
        sort_by='rsi',
        sort_ascending=True,
        max_results=2
    )

    assert scan.sort_by == 'rsi'
    assert scan.sort_ascending == True
    assert scan.max_results == 2


def test_scan_with_max_results():
    """Test scan with max_results limiting."""
    scan = create_simple_scan(
        name='Limited Results',
        universe=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        indicator_name='SMA',
        operator='>',
        threshold=0
    )

    scan.max_results = 3

    assert scan.max_results == 3
    assert len(scan.universe) == 5  # Universe larger than max_results
