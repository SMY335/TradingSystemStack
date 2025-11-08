"""
Core indicator execution engine.
"""
from typing import Dict, Any, Optional
import pandas as pd
import logging

from src.utils.registry import _indicator_registry as indicators_registry
from .exceptions import IndicatorNotFoundError, CalculationError
from .config import get_config

logger = logging.getLogger(__name__)


def run_indicator(
    indicator_name: str,
    df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    library: Optional[str] = None
) -> pd.DataFrame:
    """Run indicator dynamically from registry.

    Args:
        indicator_name: Name of indicator (e.g., 'RSI', 'MACD', 'EMA')
        df: Input DataFrame with OHLCV data
        params: Indicator parameters
        library: Preferred library (None = auto-detect)

    Returns:
        DataFrame with indicator columns added

    Raises:
        IndicatorNotFoundError: If indicator not in registry
        CalculationError: If calculation fails

    Examples:
        >>> result = run_indicator('RSI', df, params={'length': 14})
        >>> result = run_indicator('MACD', df, library='talib')
    """
    if params is None:
        params = {}

    config = get_config()

    # Normalize indicator name
    indicator_key = indicator_name.upper()

    # Try to find indicator in registry
    indicator_class = None

    # Use a sentinel to detect when default should be used
    _NOTFOUND = object()
    try:
        indicator_class = indicators_registry.get(indicator_key, default=_NOTFOUND)
        if indicator_class is _NOTFOUND:
            indicator_class = None
    except Exception:
        indicator_class = None

    if indicator_class is None:
        # Try aliases
        for key in indicators_registry.list_all():
            entry = indicators_registry.get_entry(key)
            if entry and indicator_key in [a.upper() for a in entry.aliases]:
                indicator_class = entry.obj
                break

    if indicator_class is None:
        raise IndicatorNotFoundError(
            f"Indicator '{indicator_name}' not found in registry. "
            f"Available: {', '.join(indicators_registry.list_all())}"
        )

    # Instantiate and calculate
    try:
        indicator = indicator_class()
        result = indicator.calculate(df, **params)
        logger.debug(f"Calculated {indicator_name} successfully")
        return result

    except Exception as e:
        logger.error(f"Failed to calculate {indicator_name}: {e}")
        raise CalculationError(
            f"Indicator calculation failed for {indicator_name}: {e}"
        ) from e


def get_available_indicators(library: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get list of available indicators.

    Args:
        library: Filter by library (None = all)

    Returns:
        Dictionary of indicator metadata

    Examples:
        >>> indicators = get_available_indicators()
        >>> talib_indicators = get_available_indicators(library='talib')
    """
    all_indicators = {}

    for name in indicators_registry.list_all():
        entry = indicators_registry.get_entry(name)
        if entry:
            indicator_lib = entry.metadata.get('library', '') if entry.metadata else ''

            if library is None or indicator_lib == library:
                all_indicators[name] = {
                    'library': indicator_lib,
                    'aliases': entry.aliases,
                    'category': entry.category or 'unknown',
                    'params_schema': entry.params_schema or {}
                }

    return all_indicators
