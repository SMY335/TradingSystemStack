"""
TA-Lib indicator wrappers.

Wraps TA-Lib indicators with unified interface.
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from src.indicators.base import BaseIndicator
from src.indicators.exceptions import LibraryNotAvailableError, CalculationError
from src.utils.registry import register_indicator

logger = logging.getLogger(__name__)


class TALibIndicator(BaseIndicator):
    """Base class for TA-Lib indicators."""

    def __init__(self, name: str, talib_func_name: str):
        super().__init__(name, 'talib')
        self.talib_func_name = talib_func_name

        if not TALIB_AVAILABLE:
            raise LibraryNotAvailableError("TA-Lib not available. Install: pip install TA-Lib")

    def _get_talib_function(self):
        """Get TA-Lib function."""
        if not hasattr(talib, self.talib_func_name):
            raise CalculationError(f"TA-Lib function {self.talib_func_name} not found")
        return getattr(talib, self.talib_func_name)


@register_indicator('RSI', aliases=['rsi'])
class RSI(TALibIndicator):
    """Relative Strength Index."""

    def __init__(self):
        super().__init__('RSI', 'RSI')

    def calculate(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Calculate RSI.

        Args:
            df: DataFrame with 'close' column
            length: RSI period

        Returns:
            DataFrame with 'rsi' column added
        """
        self.validate_input(df, required_columns=['close'])

        result = df.copy()
        result['rsi'] = talib.RSI(df['close'].values, timeperiod=length)

        return result


@register_indicator('MACD', aliases=['macd'])
class MACD(TALibIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self):
        super().__init__('MACD', 'MACD')

    def calculate(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD.

        Args:
            df: DataFrame with 'close' column
            fast: Fast period
            slow: Slow period
            signal: Signal period

        Returns:
            DataFrame with 'macd', 'macd_signal', 'macd_hist' columns
        """
        self.validate_input(df, required_columns=['close'])

        result = df.copy()
        macd, signal_line, hist = talib.MACD(
            df['close'].values,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )

        result['macd'] = macd
        result['macd_signal'] = signal_line
        result['macd_hist'] = hist

        return result


@register_indicator('EMA', aliases=['ema'])
class EMA(TALibIndicator):
    """Exponential Moving Average."""

    def __init__(self):
        super().__init__('EMA', 'EMA')

    def calculate(self, df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
        """Calculate EMA.

        Args:
            df: DataFrame with 'close' column
            length: EMA period

        Returns:
            DataFrame with 'ema' column added
        """
        self.validate_input(df, required_columns=['close'])

        result = df.copy()
        result[f'ema_{length}'] = talib.EMA(df['close'].values, timeperiod=length)

        return result


@register_indicator('SMA', aliases=['sma'])
class SMA(TALibIndicator):
    """Simple Moving Average."""

    def __init__(self):
        super().__init__('SMA', 'SMA')

    def calculate(self, df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
        """Calculate SMA."""
        self.validate_input(df, required_columns=['close'])

        result = df.copy()
        result[f'sma_{length}'] = talib.SMA(df['close'].values, timeperiod=length)

        return result


@register_indicator('BBANDS', aliases=['bollinger', 'bb'])
class BBANDS(TALibIndicator):
    """Bollinger Bands."""

    def __init__(self):
        super().__init__('BBANDS', 'BBANDS')

    def calculate(
        self,
        df: pd.DataFrame,
        length: int = 20,
        std: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'close' column
            length: Period
            std: Standard deviation multiplier

        Returns:
            DataFrame with 'bb_upper', 'bb_middle', 'bb_lower' columns
        """
        self.validate_input(df, required_columns=['close'])

        result = df.copy()
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=length,
            nbdevup=std,
            nbdevdn=std
        )

        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower

        return result


@register_indicator('ATR', aliases=['atr'])
class ATR(TALibIndicator):
    """Average True Range."""

    def __init__(self):
        super().__init__('ATR', 'ATR')

    def calculate(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Calculate ATR."""
        self.validate_input(df, required_columns=['high', 'low', 'close'])

        result = df.copy()
        result['atr'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=length
        )

        return result


@register_indicator('STOCH', aliases=['stochastic'])
class STOCH(TALibIndicator):
    """Stochastic Oscillator."""

    def __init__(self):
        super().__init__('STOCH', 'STOCH')

    def calculate(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic."""
        self.validate_input(df, required_columns=['high', 'low', 'close'])

        result = df.copy()
        k, d = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=k_period,
            slowk_period=d_period,
            slowd_period=d_period
        )

        result['stoch_k'] = k
        result['stoch_d'] = d

        return result


@register_indicator('ADX', aliases=['adx'])
class ADX(TALibIndicator):
    """Average Directional Index."""

    def __init__(self):
        super().__init__('ADX', 'ADX')

    def calculate(self, df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
        """Calculate ADX."""
        self.validate_input(df, required_columns=['high', 'low', 'close'])

        result = df.copy()
        result['adx'] = talib.ADX(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=length
        )

        return result


def is_talib_available() -> bool:
    """Check if TA-Lib is available."""
    return TALIB_AVAILABLE
