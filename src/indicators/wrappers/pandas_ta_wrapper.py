"""
Pandas-TA indicator wrappers (fallback when TA-Lib not available).
"""
from typing import Dict, Any
import pandas as pd
import logging

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from src.indicators.base import BaseIndicator
from src.indicators.exceptions import LibraryNotAvailableError
from src.utils.registry import register_indicator

logger = logging.getLogger(__name__)


class PandasTAIndicator(BaseIndicator):
    """Base class for pandas-ta indicators."""

    def __init__(self, name: str):
        super().__init__(name, 'pandas_ta')

        if not PANDAS_TA_AVAILABLE:
            raise LibraryNotAvailableError(
                "pandas-ta not available. Install: pip install pandas-ta"
            )


@register_indicator('SUPERTREND', aliases=['supertrend'])
class SuperTrend(PandasTAIndicator):
    """SuperTrend indicator."""

    def __init__(self):
        super().__init__('SUPERTREND')

    def calculate(
        self,
        df: pd.DataFrame,
        length: int = 10,
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """Calculate SuperTrend.

        Args:
            df: DataFrame with OHLC data
            length: ATR period
            multiplier: ATR multiplier

        Returns:
            DataFrame with 'supertrend', 'supertrend_direction' columns
        """
        self.validate_input(df, required_columns=['high', 'low', 'close'])

        result = df.copy()

        if PANDAS_TA_AVAILABLE:
            st = pta.supertrend(
                df['high'],
                df['low'],
                df['close'],
                length=length,
                multiplier=multiplier
            )

            result['supertrend'] = st[f'SUPERT_{length}_{multiplier}']
            result['supertrend_direction'] = st[f'SUPERTd_{length}_{multiplier}']

        return result


@register_indicator('ICHIMOKU', aliases=['ichimoku'])
class Ichimoku(PandasTAIndicator):
    """Ichimoku Cloud."""

    def __init__(self):
        super().__init__('ICHIMOKU')

    def calculate(
        self,
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou: int = 52
    ) -> pd.DataFrame:
        """Calculate Ichimoku.

        Args:
            df: DataFrame with OHLC data
            tenkan: Tenkan-sen period
            kijun: Kijun-sen period
            senkou: Senkou span period

        Returns:
            DataFrame with Ichimoku columns
        """
        self.validate_input(df, required_columns=['high', 'low', 'close'])

        result = df.copy()

        if PANDAS_TA_AVAILABLE:
            ich = pta.ichimoku(df['high'], df['low'], df['close'])[0]

            result['tenkan_sen'] = ich[f'ITS_{tenkan}']
            result['kijun_sen'] = ich[f'IKS_{kijun}']
            result['senkou_span_a'] = ich[f'ISA_{tenkan}']
            result['senkou_span_b'] = ich[f'ISB_{kijun}']

        return result


def is_pandas_ta_available() -> bool:
    """Check if pandas-ta is available."""
    return PANDAS_TA_AVAILABLE
