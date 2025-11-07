"""
Data ingestion module for TradingSystemStack.

Provides unified data loading from multiple sources with automatic
normalization, caching, and validation.

Supported Sources:
- yfinance: Stock market data
- CCXT: Cryptocurrency exchange data
- FRED: Economic indicators
- CSV/Parquet: Local files

Examples:
    >>> from src.data import get_ohlcv, get_series, get_cache
    >>>
    >>> # Load stock data
    >>> df = get_ohlcv('AAPL', start='2023-01-01')
    >>>
    >>> # Load crypto data
    >>> df = get_ohlcv('BTC/USDT', source='ccxt')
    >>>
    >>> # Load economic data
    >>> gdp = get_series('GDP')
    >>>
    >>> # Use caching
    >>> cache = get_cache()
    >>> df = cache.get_or_load('AAPL_1d', lambda: get_ohlcv('AAPL'))
"""

# Data loaders
from .loaders import (
    get_ohlcv,
    load_yfinance,
    load_ccxt,
    load_csv,
    load_parquet,
    save_ohlcv,
    get_available_sources,
    DataLoaderError,
)

# Normalizers
from .normalizers import (
    normalize_columns,
    normalize_ohlcv,
    validate_ohlcv,
    ensure_numeric_types,
    remove_invalid_rows,
    ensure_sorted,
    remove_duplicates,
    detect_data_issues,
    get_required_columns,
    get_column_aliases,
)

# FRED economic data
from .fred import (
    get_series,
    get_multiple_series,
    get_popular_series,
    is_fred_available,
    FREDClient,
    FREDError,
)

# Cache
from .cache import (
    DataCache,
    get_cache,
    CacheError,
)


__all__ = [
    # Loaders
    'get_ohlcv',
    'load_yfinance',
    'load_ccxt',
    'load_csv',
    'load_parquet',
    'save_ohlcv',
    'get_available_sources',
    'DataLoaderError',

    # Normalizers
    'normalize_columns',
    'normalize_ohlcv',
    'validate_ohlcv',
    'ensure_numeric_types',
    'remove_invalid_rows',
    'ensure_sorted',
    'remove_duplicates',
    'detect_data_issues',
    'get_required_columns',
    'get_column_aliases',

    # FRED
    'get_series',
    'get_multiple_series',
    'get_popular_series',
    'is_fred_available',
    'FREDClient',
    'FREDError',

    # Cache
    'DataCache',
    'get_cache',
    'CacheError',
]
