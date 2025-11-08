"""
Base indicator class and utilities.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from .validators import validate_dataframe, validate_parameters
from .exceptions import CalculationError

logger = logging.getLogger(__name__)


class BaseIndicator(ABC):
    """Abstract base class for all indicators.

    All indicator wrappers should inherit from this class.
    """

    def __init__(self, name: str, library: str):
        """Initialize indicator.

        Args:
            name: Indicator name
            library: Source library (talib, pandas_ta, etc.)
        """
        self.name = name
        self.library = library

    @abstractmethod
    def calculate(
        self,
        df: pd.DataFrame,
        **params
    ) -> pd.DataFrame:
        """Calculate indicator.

        Args:
            df: Input DataFrame with OHLCV data
            **params: Indicator-specific parameters

        Returns:
            DataFrame with indicator columns added

        Raises:
            CalculationError: If calculation fails
        """
        pass

    def validate_input(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> None:
        """Validate input DataFrame.

        Args:
            df: DataFrame to validate
            required_columns: Required columns

        Raises:
            InvalidDataError: If validation fails
        """
        if required_columns is None:
            required_columns = ['close']  # Default minimum

        validate_dataframe(df, required_columns=required_columns)

    def validate_params(
        self,
        params: Dict[str, Any],
        schema: Optional[Dict[str, type]] = None
    ) -> None:
        """Validate parameters.

        Args:
            params: Parameters to validate
            schema: Parameter type schema

        Raises:
            InvalidParametersError: If validation fails
        """
        validate_parameters(params, schema=schema)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', library='{self.library}')"
