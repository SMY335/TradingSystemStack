"""
Validation utilities for indicators.
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

from .exceptions import InvalidDataError, InvalidParametersError

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> None:
    """Validate DataFrame for indicator calculation.

    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present
        min_rows: Minimum number of rows required

    Raises:
        InvalidDataError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataError("Input must be a pandas DataFrame")

    if df.empty:
        raise InvalidDataError("DataFrame is empty")

    if len(df) < min_rows:
        raise InvalidDataError(
            f"DataFrame has {len(df)} rows, minimum {min_rows} required"
        )

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise InvalidDataError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )


def validate_ohlcv(df: pd.DataFrame, require_volume: bool = True) -> None:
    """Validate DataFrame has OHLCV columns.

    Args:
        df: DataFrame to validate
        require_volume: Require volume column

    Raises:
        InvalidDataError: If OHLCV columns missing
    """
    required = ['open', 'high', 'low', 'close']
    if require_volume:
        required.append('volume')

    validate_dataframe(df, required_columns=required)


def validate_parameters(
    params: Dict[str, Any],
    required: Optional[List[str]] = None,
    schema: Optional[Dict[str, type]] = None
) -> None:
    """Validate indicator parameters.

    Args:
        params: Parameters to validate
        required: Required parameter names
        schema: Expected parameter types

    Raises:
        InvalidParametersError: If validation fails
    """
    if not isinstance(params, dict):
        raise InvalidParametersError("Parameters must be a dictionary")

    # Check required parameters
    if required:
        missing = [p for p in required if p not in params]
        if missing:
            raise InvalidParametersError(
                f"Missing required parameters: {missing}"
            )

    # Check parameter types
    if schema:
        for param_name, expected_type in schema.items():
            if param_name in params:
                value = params[param_name]
                if not isinstance(value, expected_type):
                    raise InvalidParametersError(
                        f"Parameter '{param_name}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
