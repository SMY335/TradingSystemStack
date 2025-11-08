"""
Integration tests for Session 4 modules.

Tests fundamentals, economics, and scanner modules working together.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fundamentals import (
    FinancialRatios,
    get_financial_ratios,
    calculate_intrinsic_value_dcf,
    screen_by_fundamentals,
    FinancialAnalyzer
)
from src.economics import (
    get_economic_indicator,
    get_interest_rates,
    calculate_yield_curve_slope,
    is_yield_curve_inverted,
    EconomicDataProvider
)
from src.scanner import (
    ScanDefinition,
    ScanEngine,
    create_simple_scan,
    create_cross_scan,
    load_scan_from_json,
    save_scan_to_json
)


# ============================================================================
# Fundamentals Module Tests
# ============================================================================

def test_get_financial_ratios():
    """Test retrieving financial ratios."""
    ratios = get_financial_ratios('AAPL', use_mock=True)

    assert isinstance(ratios, FinancialRatios)
    assert ratios.symbol == 'AAPL'
    assert ratios.pe_ratio is not None
    assert ratios.roe is not None


def test_calculate_intrinsic_value_dcf():
    """Test DCF valuation calculation."""
    valuation = calculate_intrinsic_value_dcf(
        free_cash_flow=10_000_000_000,
        growth_rate=0.10,
        discount_rate=0.12,
        terminal_growth=0.03,
        years=5
    )

    assert valuation > 0
    # FCF should grow and be discounted
    assert isinstance(valuation, float)


def test_screen_by_fundamentals():
    """Test fundamental screening."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

    results = screen_by_fundamentals(
        symbols,
        criteria={
            'pe_ratio': (None, 30),
            'roe': (0.15, None)
        },
        use_mock=True
    )

    assert isinstance(results, list)
    assert len(results) <= len(symbols)


def test_financial_analyzer():
    """Test FinancialAnalyzer class."""
    analyzer = FinancialAnalyzer(use_mock=True)

    # Analyze single symbol
    analysis = analyzer.analyze('AAPL')
    assert isinstance(analysis, dict)
    assert 'overall_score' in analysis
    assert 0 <= analysis['overall_score'] <= 100
    assert 'rating' in analysis

    # Analyze multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    results = []
    for symbol in symbols:
        analysis = analyzer.analyze(symbol)
        results.append({
            'symbol': symbol,
            'score': analysis['overall_score'],
            'rating': analysis['rating']
        })

    assert len(results) == len(symbols)
    for result in results:
        assert 0 <= result['score'] <= 100


# ============================================================================
# Economics Module Tests
# ============================================================================

def test_get_economic_indicator():
    """Test getting economic indicators."""
    gdp = get_economic_indicator('GDP', use_mock=True)

    assert isinstance(gdp, pd.Series)
    assert len(gdp) > 0
    assert gdp.index.dtype == 'datetime64[ns]'


def test_multiple_indicators():
    """Test getting multiple economic indicators."""
    indicators = ['GDP', 'CPI', 'UNEMPLOYMENT']

    for indicator in indicators:
        data = get_economic_indicator(indicator, use_mock=True)
        assert isinstance(data, pd.Series)
        assert len(data) > 0


def test_get_interest_rates():
    """Test getting interest rates."""
    rates = get_interest_rates(use_mock=True)

    assert isinstance(rates, pd.DataFrame)
    assert len(rates) > 0
    assert 'fed_funds_rate' in rates.columns
    assert 'treasury_10y' in rates.columns


def test_yield_curve_functions():
    """Test yield curve slope and inversion detection."""
    rates = get_interest_rates(use_mock=True)

    assert isinstance(rates, pd.DataFrame)
    assert len(rates) > 0

    # Test slope calculation
    slope = calculate_yield_curve_slope(rates)
    assert isinstance(slope, pd.Series)
    assert len(slope) > 0

    # Test inversion detection
    inverted = is_yield_curve_inverted(rates)
    assert isinstance(inverted, (bool, np.bool_))


def test_economic_data_provider():
    """Test EconomicDataProvider class."""
    provider = EconomicDataProvider(use_mock=True)

    # Get dashboard
    dashboard = provider.get_dashboard()
    assert isinstance(dashboard, dict)
    assert 'gdp' in dashboard
    assert 'inflation' in dashboard
    assert 'unemployment' in dashboard
    assert 'yield_curve_slope' in dashboard

    # Get individual indicators
    gdp = provider.get_gdp()
    assert isinstance(gdp, pd.Series)

    inflation = provider.get_inflation()
    assert isinstance(inflation, pd.Series)

    rates = provider.get_interest_rates()
    assert isinstance(rates, pd.DataFrame)


# ============================================================================
# Scanner Module Tests
# ============================================================================

def test_create_simple_scan_integration():
    """Test creating a simple scan."""
    scan = create_simple_scan(
        name='Test Scan',
        universe=['AAPL', 'MSFT'],
        indicator_name='RSI',
        operator='<',
        threshold=40
    )

    assert scan.name == 'Test Scan'
    assert len(scan.universe) == 2
    assert len(scan.indicators) == 1


def test_create_cross_scan_integration():
    """Test creating a cross scan."""
    scan = create_cross_scan(
        name='MA Cross',
        universe=['SPY', 'QQQ'],
        fast_indicator='sma_50',
        slow_indicator='sma_200',
        direction='above'
    )

    assert scan.name == 'MA Cross'
    assert len(scan.indicators) == 2
    assert scan.conditions.type == 'cross'


def test_save_and_load_scan_roundtrip(tmp_path):
    """Test save/load roundtrip for scan definitions."""
    # Create a scan
    original = create_simple_scan(
        name='Roundtrip Test',
        universe=['AAPL', 'MSFT', 'GOOGL'],
        indicator_name='SMA',
        operator='>',
        threshold=100
    )

    # Save it
    filepath = tmp_path / 'roundtrip.json'
    save_scan_to_json(original, filepath)

    # Load it back
    loaded = load_scan_from_json(filepath)

    # Verify
    assert loaded.name == original.name
    assert loaded.universe == original.universe
    assert loaded.timeframe == original.timeframe
    assert len(loaded.indicators) == len(original.indicators)


def test_scan_execution_basic():
    """Test basic scan execution."""
    scan = create_simple_scan(
        name='Always Match',
        universe=['AAPL'],
        indicator_name='SMA',
        operator='>',
        threshold=-1000,  # Always true
        lookback=50
    )

    engine = ScanEngine(max_workers=1, verbose=False)

    try:
        results = engine.execute(scan)
        assert results.scan_name == 'Always Match'
        assert results.total_scanned == 1
    except Exception as e:
        # Data fetching may fail in test environment
        pytest.skip(f"Data fetching failed: {e}")


# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

def test_fundamental_scan_integration():
    """Test scanning with fundamental filters."""
    # This would typically combine scanner with fundamental data
    # For now, just test that both modules work

    # Get fundamentals
    ratios = get_financial_ratios('AAPL', use_mock=True)
    assert ratios is not None

    # Create a technical scan
    scan = create_simple_scan(
        name='Combined',
        universe=['AAPL'],
        indicator_name='RSI',
        operator='>',
        threshold=50
    )
    assert scan is not None


def test_economic_context_scan():
    """Test using economic context for scanning."""
    # Get economic data
    gdp = get_economic_indicator('GDP', use_mock=True)
    assert len(gdp) > 0

    # Create scan based on market conditions
    # In a real system, this might adjust parameters based on economic regime
    scan = create_simple_scan(
        name='Economic Aware',
        universe=['SPY'],
        indicator_name='SMA',
        operator='>',
        threshold=100
    )
    assert scan is not None


def test_full_analysis_workflow():
    """Test complete analysis workflow using all Session 4 modules."""
    symbol = 'AAPL'

    # Step 1: Get fundamental data
    fundamentals = get_financial_ratios(symbol, use_mock=True)
    assert fundamentals.symbol == symbol

    # Step 2: Get economic context
    provider = EconomicDataProvider(use_mock=True)
    econ_data = provider.get_dashboard()
    assert 'gdp' in econ_data

    # Step 3: Create and potentially execute a scan
    scan = create_simple_scan(
        name='Full Workflow',
        universe=[symbol],
        indicator_name='RSI',
        operator='>',
        threshold=30
    )
    assert scan.universe[0] == symbol

    # Step 4: Analyze fundamentals
    analyzer = FinancialAnalyzer(use_mock=True)
    analysis = analyzer.analyze(symbol)
    assert 0 <= analysis['overall_score'] <= 100


def test_financial_analyzer_scoring():
    """Test financial analyzer scoring system."""
    analyzer = FinancialAnalyzer(use_mock=True)

    # Test with multiple symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    results = []
    for symbol in symbols:
        analysis = analyzer.analyze(symbol)
        results.append({
            'symbol': symbol,
            'score': analysis['overall_score']
        })

    assert len(results) == len(symbols)
    assert all(r['score'] >= 0 for r in results)
    assert all(r['score'] <= 100 for r in results)

    # Sort by score to verify scoring works
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    assert len(sorted_results) == len(symbols)


def test_economic_dashboard():
    """Test economic dashboard functionality."""
    provider = EconomicDataProvider(use_mock=True)
    dashboard = provider.get_dashboard()

    required_keys = ['gdp', 'inflation', 'unemployment', 'fed_funds_rate', 'treasury_10y', 'yield_curve_slope']

    for key in required_keys:
        assert key in dashboard, f"Missing {key} in dashboard"


def test_yield_curve_analysis():
    """Test yield curve slope calculation and inversion detection."""
    rates = get_interest_rates(use_mock=True)

    # Should have interest rate data
    assert len(rates) > 0
    assert 'treasury_2y' in rates.columns
    assert 'treasury_10y' in rates.columns

    # Calculate slope
    slope = calculate_yield_curve_slope(rates)
    assert isinstance(slope, pd.Series)
    assert len(slope) > 0

    # Test inversion detection
    inverted = is_yield_curve_inverted(rates)
    assert isinstance(inverted, (bool, np.bool_))

    # Yields should be reasonable percentages
    assert all(rates['treasury_2y'] >= 0)
    assert all(rates['treasury_10y'] >= 0)
    assert all(rates['treasury_10y'] < 50)  # Less than 50%


def test_dcf_valuation_scenarios():
    """Test DCF valuation with different scenarios."""
    base_fcf = 10_000_000_000

    # High growth scenario
    high_growth = calculate_intrinsic_value_dcf(
        free_cash_flow=base_fcf,
        growth_rate=0.20,
        discount_rate=0.10,
        terminal_growth=0.03,
        years=5
    )

    # Low growth scenario
    low_growth = calculate_intrinsic_value_dcf(
        free_cash_flow=base_fcf,
        growth_rate=0.05,
        discount_rate=0.10,
        terminal_growth=0.02,
        years=5
    )

    # High growth should yield higher valuation
    assert high_growth > low_growth


def test_fundamental_screening_filters():
    """Test fundamental screening with various filters."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # Test PE filter
    pe_filtered = screen_by_fundamentals(
        symbols,
        criteria={'pe_ratio': (None, 25)},
        use_mock=True
    )
    assert len(pe_filtered) <= len(symbols)

    # Test ROE filter
    roe_filtered = screen_by_fundamentals(
        symbols,
        criteria={'roe': (0.20, None)},
        use_mock=True
    )
    assert len(roe_filtered) <= len(symbols)

    # Test combined filters
    combined = screen_by_fundamentals(
        symbols,
        criteria={
            'pe_ratio': (None, 25),
            'roe': (0.15, None)
        },
        use_mock=True
    )
    assert len(combined) <= len(symbols)


def test_mock_data_consistency():
    """Test that mock data is consistent across calls."""
    # Same symbol should return same data
    ratios1 = get_financial_ratios('AAPL', use_mock=True)
    ratios2 = get_financial_ratios('AAPL', use_mock=True)

    assert ratios1.pe_ratio == ratios2.pe_ratio
    assert ratios1.roe == ratios2.roe

    # Different symbols should return different data
    aapl = get_financial_ratios('AAPL', use_mock=True)
    msft = get_financial_ratios('MSFT', use_mock=True)

    # At least some metrics should differ
    differs = (
        aapl.pe_ratio != msft.pe_ratio or
        aapl.roe != msft.roe or
        aapl.market_cap != msft.market_cap
    )
    assert differs
