"""
Anchored VWAP (Volume Weighted Average Price) calculation.

Provides VWAP calculations anchored to specific points in time.
"""
from typing import Union, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from .anchors import (
    get_anchor,
    timestamp_anchor,
    custom_anchor,
    AnchorError,
)

logger = logging.getLogger(__name__)


class AnchoredVWAP:
    """Anchored VWAP calculator.

    Calculates VWAP anchored to specific points (session, week, swing, etc.).

    Examples:
        >>> vwap_calc = AnchoredVWAP()
        >>> vwap = vwap_calc.calculate(df, anchor_type='session')
        >>> vwap = vwap_calc.calculate(df, anchor_type='swing_high')
    """

    def __init__(self, include_bands: bool = True, num_std: float = 1.0):
        """Initialize VWAP calculator.

        Args:
            include_bands: Include upper/lower bands (default: True)
            num_std: Number of standard deviations for bands (default: 1.0)
        """
        self.include_bands = include_bands
        self.num_std = num_std

    def calculate(
        self,
        df: pd.DataFrame,
        anchor_type: Optional[str] = None,
        anchor_timestamp: Optional[Union[str, datetime]] = None,
        anchor_func: Optional[Callable] = None,
        price_col: str = 'close',
        **anchor_kwargs
    ) -> pd.DataFrame:
        """Calculate anchored VWAP.

        Args:
            df: DataFrame with OHLC and volume data
            anchor_type: Anchor type ('session', 'week', 'month', 'swing_high', etc.)
            anchor_timestamp: Specific timestamp to anchor
            anchor_func: Custom anchor function
            price_col: Price column to use (default: 'close')
            **anchor_kwargs: Additional arguments for anchor function

        Returns:
            DataFrame with VWAP and optional bands

        Examples:
            >>> result = vwap_calc.calculate(df, anchor_type='session')
            >>> result = vwap_calc.calculate(df, anchor_timestamp='2024-01-01')
            >>> result = vwap_calc.calculate(df, anchor_func=my_anchor_func)
        """
        if 'volume' not in df.columns:
            raise ValueError("DataFrame must have 'volume' column")

        if price_col not in df.columns:
            raise ValueError(f"DataFrame must have '{price_col}' column")

        # Get anchor points
        if anchor_timestamp is not None:
            anchors = timestamp_anchor(df, anchor_timestamp)
        elif anchor_func is not None:
            anchors = custom_anchor(df, anchor_func)
        elif anchor_type is not None:
            anchors = get_anchor(df, anchor_type, **anchor_kwargs)
        else:
            # Default: anchor at first bar
            anchors = pd.Series(False, index=df.index)
            anchors.iloc[0] = True

        # Calculate VWAP for each anchor period
        vwap = self._calculate_vwap_periods(df, anchors, price_col)

        result = pd.DataFrame({'vwap': vwap}, index=df.index)

        # Add bands if requested
        if self.include_bands:
            upper, lower = self._calculate_bands(df, anchors, vwap, price_col)
            result['vwap_upper'] = upper
            result['vwap_lower'] = lower

        return result

    def _calculate_vwap_periods(
        self,
        df: pd.DataFrame,
        anchors: pd.Series,
        price_col: str
    ) -> pd.Series:
        """Calculate VWAP across multiple anchor periods."""
        prices = df[price_col].values
        volumes = df['volume'].values

        # Initialize VWAP array
        vwap = np.zeros(len(df))

        # Cumulative price * volume and cumulative volume
        cum_pv = np.zeros(len(df))
        cum_v = np.zeros(len(df))

        # Track current period start
        period_start = 0

        for i in range(len(df)):
            # Check if new anchor
            if anchors.iloc[i]:
                period_start = i
                cum_pv[i] = prices[i] * volumes[i]
                cum_v[i] = volumes[i]
            else:
                # Continue current period
                if i == 0:
                    cum_pv[i] = prices[i] * volumes[i]
                    cum_v[i] = volumes[i]
                else:
                    cum_pv[i] = cum_pv[i-1] + prices[i] * volumes[i]
                    cum_v[i] = cum_v[i-1] + volumes[i]

            # Calculate VWAP
            if cum_v[i] > 0:
                vwap[i] = cum_pv[i] / cum_v[i]
            else:
                vwap[i] = prices[i]

        return pd.Series(vwap, index=df.index)

    def _calculate_bands(
        self,
        df: pd.DataFrame,
        anchors: pd.Series,
        vwap: pd.Series,
        price_col: str
    ) -> tuple:
        """Calculate VWAP bands based on standard deviation."""
        prices = df[price_col].values
        volumes = df['volume'].values

        # Calculate variance for each period
        variance = np.zeros(len(df))
        cum_pv_squared = np.zeros(len(df))
        cum_v = np.zeros(len(df))

        for i in range(len(df)):
            if anchors.iloc[i]:
                # New period
                cum_pv_squared[i] = (prices[i] ** 2) * volumes[i]
                cum_v[i] = volumes[i]
            else:
                if i == 0:
                    cum_pv_squared[i] = (prices[i] ** 2) * volumes[i]
                    cum_v[i] = volumes[i]
                else:
                    cum_pv_squared[i] = cum_pv_squared[i-1] + (prices[i] ** 2) * volumes[i]
                    cum_v[i] = cum_v[i-1] + volumes[i]

            # Calculate variance
            if cum_v[i] > 0:
                avg_price_squared = cum_pv_squared[i] / cum_v[i]
                variance[i] = avg_price_squared - (vwap.iloc[i] ** 2)
                variance[i] = max(variance[i], 0)  # Handle numerical errors
            else:
                variance[i] = 0

        # Standard deviation
        std = np.sqrt(variance)

        # Bands
        upper = vwap + (std * self.num_std)
        lower = vwap - (std * self.num_std)

        return (
            pd.Series(upper, index=df.index),
            pd.Series(lower, index=df.index)
        )


def calculate_vwap(
    df: pd.DataFrame,
    anchor_type: Optional[str] = 'session',
    include_bands: bool = True,
    num_std: float = 1.0,
    **kwargs
) -> pd.DataFrame:
    """Calculate anchored VWAP (convenience function).

    Args:
        df: DataFrame with OHLC and volume data
        anchor_type: Anchor type ('session', 'week', 'month', 'swing_high', etc.)
        include_bands: Include upper/lower bands
        num_std: Number of standard deviations for bands
        **kwargs: Additional arguments for anchor function

    Returns:
        DataFrame with VWAP and optional bands

    Examples:
        >>> vwap_df = calculate_vwap(df, anchor_type='session')
        >>> vwap_df = calculate_vwap(df, anchor_type='swing_high', lookback=20)
    """
    calculator = AnchoredVWAP(include_bands=include_bands, num_std=num_std)
    return calculator.calculate(df, anchor_type=anchor_type, **kwargs)


def calculate_vwap_from_timestamp(
    df: pd.DataFrame,
    timestamp: Union[str, datetime],
    include_bands: bool = True,
    num_std: float = 1.0
) -> pd.DataFrame:
    """Calculate VWAP anchored to specific timestamp.

    Args:
        df: DataFrame with OHLC and volume data
        timestamp: Timestamp to anchor
        include_bands: Include upper/lower bands
        num_std: Number of standard deviations for bands

    Returns:
        DataFrame with VWAP and optional bands

    Examples:
        >>> vwap_df = calculate_vwap_from_timestamp(df, '2024-01-01')
    """
    calculator = AnchoredVWAP(include_bands=include_bands, num_std=num_std)
    return calculator.calculate(df, anchor_timestamp=timestamp)


def calculate_multiple_vwaps(
    df: pd.DataFrame,
    anchor_types: list,
    include_bands: bool = False
) -> pd.DataFrame:
    """Calculate multiple VWAPs with different anchors.

    Args:
        df: DataFrame with OHLC and volume data
        anchor_types: List of anchor types
        include_bands: Include upper/lower bands for each

    Returns:
        DataFrame with VWAP for each anchor type

    Examples:
        >>> vwaps = calculate_multiple_vwaps(df, ['session', 'week', 'month'])
    """
    calculator = AnchoredVWAP(include_bands=include_bands)

    results = {}
    for anchor_type in anchor_types:
        try:
            vwap_df = calculator.calculate(df, anchor_type=anchor_type)

            if include_bands:
                results[f'vwap_{anchor_type}'] = vwap_df['vwap']
                results[f'vwap_{anchor_type}_upper'] = vwap_df['vwap_upper']
                results[f'vwap_{anchor_type}_lower'] = vwap_df['vwap_lower']
            else:
                results[f'vwap_{anchor_type}'] = vwap_df['vwap']

        except Exception as e:
            logger.warning(f"Failed to calculate VWAP for {anchor_type}: {e}")

    return pd.DataFrame(results, index=df.index)
