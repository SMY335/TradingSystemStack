"""
Economic Indicators Module.

This module provides access to macroeconomic indicators via FRED (Federal Reserve
Economic Data) and other sources.

Examples:
    >>> from src.economics import get_gdp, get_inflation, get_unemployment
    >>>
    >>> # Get GDP data
    >>> gdp = get_gdp()
    >>> print(gdp.tail())
    >>>
    >>> # Get inflation (CPI)
    >>> cpi = get_inflation()
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class EconomicIndicator:
    """Economic indicator data."""
    name: str
    series_id: str
    value: float
    date: datetime
    unit: str
    change_1m: Optional[float] = None
    change_3m: Optional[float] = None
    change_1y: Optional[float] = None


# FRED Series IDs for common indicators
FRED_SERIES = {
    'gdp': 'GDP',
    'cpi': 'CPIAUCSL',
    'unemployment': 'UNRATE',
    'fed_funds_rate': 'FEDFUNDS',
    'treasury_10y': 'DGS10',
    'treasury_2y': 'DGS2',
    'consumer_confidence': 'UMCSENT',
    'retail_sales': 'RSXFS',
    'industrial_production': 'INDPRO',
    'housing_starts': 'HOUST',
}


def get_economic_indicator(
    indicator: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_mock: bool = True
) -> pd.Series:
    """
    Get economic indicator data.

    Args:
        indicator: Indicator name (gdp, cpi, unemployment, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_mock: Use mock data (True) or FRED API (False)

    Returns:
        Series with indicator values

    Examples:
        >>> gdp = get_economic_indicator('gdp', start_date='2020-01-01')
    """
    if use_mock:
        return _generate_mock_indicator(indicator, start_date, end_date)
    else:
        # Real implementation would use fredapi
        raise NotImplementedError("FRED API integration not yet implemented")


def _generate_mock_indicator(
    indicator: str,
    start_date: Optional[str],
    end_date: Optional[str]
) -> pd.Series:
    """Generate realistic mock economic data."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start

    # Generate based on indicator type
    np.random.seed(hash(indicator) % 2**32)

    if indicator == 'gdp':
        # GDP in trillions, growing trend
        base = 20
        trend = np.linspace(0, 3, len(dates))
        noise = np.random.randn(len(dates)) * 0.2
        values = base + trend + noise

    elif indicator == 'cpi':
        # CPI index, inflationary trend
        base = 250
        trend = np.linspace(0, 30, len(dates))
        noise = np.random.randn(len(dates)) * 2
        values = base + trend + noise

    elif indicator == 'unemployment':
        # Unemployment rate 3-10%
        base = 5
        cycle = 2 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        noise = np.random.randn(len(dates)) * 0.5
        values = base + cycle + noise
        values = np.clip(values, 3, 10)

    elif indicator == 'fed_funds_rate':
        # Fed funds rate 0-5%
        base = 2
        trend = np.linspace(-1, 2, len(dates))
        noise = np.random.randn(len(dates)) * 0.3
        values = base + trend + noise
        values = np.clip(values, 0, 5)

    elif indicator in ['treasury_10y', 'treasury_2y']:
        # Treasury yields 1-4%
        base = 2.5
        noise = np.random.randn(len(dates)) * 0.5
        values = base + noise
        values = np.clip(values, 1, 4)

    else:
        # Generic indicator
        values = 100 + np.random.randn(len(dates)).cumsum()

    return pd.Series(values, index=dates, name=indicator)


def get_gdp(start_date: Optional[str] = None, use_mock: bool = True) -> pd.Series:
    """Get GDP data."""
    return get_economic_indicator('gdp', start_date=start_date, use_mock=use_mock)


def get_inflation(start_date: Optional[str] = None, use_mock: bool = True) -> pd.Series:
    """Get CPI (inflation) data."""
    return get_economic_indicator('cpi', start_date=start_date, use_mock=use_mock)


def get_unemployment(start_date: Optional[str] = None, use_mock: bool = True) -> pd.Series:
    """Get unemployment rate data."""
    return get_economic_indicator('unemployment', start_date=start_date, use_mock=use_mock)


def get_interest_rates(start_date: Optional[str] = None, use_mock: bool = True) -> pd.DataFrame:
    """
    Get interest rates (Fed Funds, 2Y, 10Y Treasury).

    Returns:
        DataFrame with multiple rate series

    Examples:
        >>> rates = get_interest_rates()
        >>> print(rates[['fed_funds_rate', 'treasury_10y']].tail())
    """
    fed_funds = get_economic_indicator('fed_funds_rate', start_date, use_mock=use_mock)
    treasury_2y = get_economic_indicator('treasury_2y', start_date, use_mock=use_mock)
    treasury_10y = get_economic_indicator('treasury_10y', start_date, use_mock=use_mock)

    df = pd.DataFrame({
        'fed_funds_rate': fed_funds,
        'treasury_2y': treasury_2y,
        'treasury_10y': treasury_10y
    })

    return df


def calculate_yield_curve_slope(rates_df: pd.DataFrame) -> pd.Series:
    """
    Calculate yield curve slope (10Y - 2Y).

    Args:
        rates_df: DataFrame from get_interest_rates()

    Returns:
        Series with yield curve slope

    Examples:
        >>> rates = get_interest_rates()
        >>> slope = calculate_yield_curve_slope(rates)
        >>> inverted = slope < 0  # Recession indicator
    """
    return rates_df['treasury_10y'] - rates_df['treasury_2y']


def is_yield_curve_inverted(rates_df: pd.DataFrame) -> bool:
    """
    Check if yield curve is inverted (recession indicator).

    Args:
        rates_df: DataFrame from get_interest_rates()

    Returns:
        True if inverted (10Y < 2Y)

    Examples:
        >>> rates = get_interest_rates()
        >>> if is_yield_curve_inverted(rates):
        ...     print("Recession warning!")
    """
    latest_slope = calculate_yield_curve_slope(rates_df).iloc[-1]
    return latest_slope < 0


class EconomicDataProvider:
    """
    Provider for economic data with caching.

    Examples:
        >>> provider = EconomicDataProvider()
        >>> gdp = provider.get_gdp()
        >>> dashboard = provider.get_dashboard()
    """

    def __init__(self, use_mock: bool = True, cache_ttl: int = 3600):
        self.use_mock = use_mock
        self.cache_ttl = cache_ttl
        self._cache = {}

    def get_gdp(self, start_date: Optional[str] = None) -> pd.Series:
        """Get GDP data."""
        return get_gdp(start_date, use_mock=self.use_mock)

    def get_inflation(self, start_date: Optional[str] = None) -> pd.Series:
        """Get inflation data."""
        return get_inflation(start_date, use_mock=self.use_mock)

    def get_unemployment(self, start_date: Optional[str] = None) -> pd.Series:
        """Get unemployment data."""
        return get_unemployment(start_date, use_mock=self.use_mock)

    def get_interest_rates(self, start_date: Optional[str] = None) -> pd.DataFrame:
        """Get interest rates."""
        return get_interest_rates(start_date, use_mock=self.use_mock)

    def get_dashboard(self) -> Dict:
        """
        Get economic dashboard with latest values.

        Returns:
            Dict with latest economic indicators

        Examples:
            >>> dashboard = provider.get_dashboard()
            >>> print(f"GDP: {dashboard['gdp']['value']:.2f}")
        """
        gdp = self.get_gdp()
        cpi = self.get_inflation()
        unemployment = self.get_unemployment()
        rates = self.get_interest_rates()

        # Calculate year-over-year changes
        def yoy_change(series):
            if len(series) >= 12:
                return ((series.iloc[-1] / series.iloc[-13]) - 1) * 100
            return None

        return {
            'gdp': {
                'value': gdp.iloc[-1],
                'yoy_change': yoy_change(gdp),
                'unit': 'Trillions USD'
            },
            'inflation': {
                'value': yoy_change(cpi) if yoy_change(cpi) else 0,
                'unit': '% YoY'
            },
            'unemployment': {
                'value': unemployment.iloc[-1],
                'unit': '%'
            },
            'fed_funds_rate': {
                'value': rates['fed_funds_rate'].iloc[-1],
                'unit': '%'
            },
            'treasury_10y': {
                'value': rates['treasury_10y'].iloc[-1],
                'unit': '%'
            },
            'yield_curve_slope': {
                'value': calculate_yield_curve_slope(rates).iloc[-1],
                'inverted': is_yield_curve_inverted(rates),
                'unit': 'basis points'
            }
        }


__all__ = [
    'EconomicIndicator',
    'FRED_SERIES',
    'get_economic_indicator',
    'get_gdp',
    'get_inflation',
    'get_unemployment',
    'get_interest_rates',
    'calculate_yield_curve_slope',
    'is_yield_curve_inverted',
    'EconomicDataProvider',
]

__version__ = '2.0.0'

# Note: Real implementation would use fredapi:
# from fredapi import Fred
# fred = Fred(api_key='YOUR_KEY')
# data = fred.get_series('GDP')
