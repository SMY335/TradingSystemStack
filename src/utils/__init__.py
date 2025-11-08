"""
Utilities module for TradingSystemStack.

Provides core utilities: types, I/O, timeframes, registry, and logging.
"""

# Type definitions
from .types import (
    OHLCVData,
    UniverseData,
    IndicatorParams,
    StrategyParams,
    BacktestResult,
    ScanResult,
    Pattern,
    Trendline,
    Zone,
    SentimentScore,
    Indicator,
    Strategy,
    DataLoader,
    TimeFrame,
    Symbol,
    Price,
    Volume,
    Timestamp,
    DateRange,
)

# I/O operations
from .io import (
    read_parquet,
    write_parquet,
    read_csv,
    write_csv,
    ensure_directory,
    list_files,
    file_size_mb,
    validate_ohlcv_columns,
    IOError,
)

# Timeframe utilities
from .timeframes import (
    normalize_timeframe,
    to_pandas_freq,
    to_seconds,
    get_timeframe_name,
    resample_ohlcv,
    get_supported_timeframes,
    is_valid_timeframe,
    compare_timeframes,
    calculate_bars_needed,
    TimeframeError,
)

# Registry system
from .registry import (
    Registry,
    RegistryEntry,
    RegistryError,
    get_indicator_registry,
    get_strategy_registry,
    get_pattern_registry,
    get_operator_registry,
    get_loader_registry,
    register_indicator,
    register_strategy,
)

# Logging configuration
from .logging_config import (
    setup_logging,
    get_logger,
    set_level,
    disable_module_logging,
    enable_debug_for_module,
    LogContext,
    log_function_call,
    silence_noisy_loggers,
)

__all__ = [
    # Types
    'OHLCVData',
    'UniverseData',
    'IndicatorParams',
    'StrategyParams',
    'BacktestResult',
    'ScanResult',
    'Pattern',
    'Trendline',
    'Zone',
    'SentimentScore',
    'Indicator',
    'Strategy',
    'DataLoader',
    'TimeFrame',
    'Symbol',
    'Price',
    'Volume',
    'Timestamp',
    'DateRange',

    # I/O
    'read_parquet',
    'write_parquet',
    'read_csv',
    'write_csv',
    'ensure_directory',
    'list_files',
    'file_size_mb',
    'validate_ohlcv_columns',
    'IOError',

    # Timeframes
    'normalize_timeframe',
    'to_pandas_freq',
    'to_seconds',
    'get_timeframe_name',
    'resample_ohlcv',
    'get_supported_timeframes',
    'is_valid_timeframe',
    'compare_timeframes',
    'calculate_bars_needed',
    'TimeframeError',

    # Registry
    'Registry',
    'RegistryEntry',
    'RegistryError',
    'get_indicator_registry',
    'get_strategy_registry',
    'get_pattern_registry',
    'get_operator_registry',
    'get_loader_registry',
    'register_indicator',
    'register_strategy',

    # Logging
    'setup_logging',
    'get_logger',
    'set_level',
    'disable_module_logging',
    'enable_debug_for_module',
    'LogContext',
    'log_function_call',
    'silence_noisy_loggers',
]
