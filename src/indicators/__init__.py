"""
Unified indicator system for TradingSystemStack.

Provides dynamic indicator execution with automatic fallback between
TA-Lib, pandas-ta, and other libraries.

Examples:
    >>> from src.indicators import run_indicator, get_available_indicators
    >>>
    >>> # Run indicator dynamically
    >>> result = run_indicator('RSI', df, params={'length': 14})
    >>> result = run_indicator('MACD', df, params={'fast': 12, 'slow': 26, 'signal': 9})
    >>>
    >>> # List available indicators
    >>> indicators = get_available_indicators()
    >>> talib_indicators = get_available_indicators(library='talib')
"""

from .base import BaseIndicator
from .config import IndicatorConfig, get_config, set_config
from .core import run_indicator, get_available_indicators
from .exceptions import (
    IndicatorError,
    IndicatorNotFoundError,
    InvalidParametersError,
    InvalidDataError,
    CalculationError,
    LibraryNotAvailableError,
)
from .validators import (
    validate_dataframe,
    validate_ohlcv,
    validate_parameters,
)

# Import wrappers to trigger registration
from . import wrappers

__all__ = [
    # Core
    'BaseIndicator',
    'run_indicator',
    'get_available_indicators',

    # Config
    'IndicatorConfig',
    'get_config',
    'set_config',

    # Exceptions
    'IndicatorError',
    'IndicatorNotFoundError',
    'InvalidParametersError',
    'InvalidDataError',
    'CalculationError',
    'LibraryNotAvailableError',

    # Validators
    'validate_dataframe',
    'validate_ohlcv',
    'validate_parameters',
]
