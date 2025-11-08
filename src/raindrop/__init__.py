"""
Raindrop Chart Visualization Module.

This module provides raindrop chart visualization with volume profile,
combining candlestick data with intrabar volume distribution.

Examples:
    >>> from src.raindrop import create_raindrop_chart
    >>> import plotly.graph_objects as go
    >>>
    >>> fig = create_raindrop_chart(df)
    >>> fig.show()
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def calculate_volume_profile(
    df: pd.DataFrame,
    price_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate volume profile for price data.

    Args:
        df: DataFrame with OHLCV data
        price_bins: Number of price bins

    Returns:
        Tuple of (price_levels, volume_up, volume_down)

    Examples:
        >>> prices, vol_up, vol_down = calculate_volume_profile(df)
    """
    # Get price range
    price_min = df['low'].min()
    price_max = df['high'].max()

    # Create price bins
    price_levels = np.linspace(price_min, price_max, price_bins)
    volume_up = np.zeros(price_bins - 1)
    volume_down = np.zeros(price_bins - 1)

    # Distribute volume across price levels
    for i, row in df.iterrows():
        # Determine if bar is up or down
        is_up = row['close'] >= row['open']

        # Find which price bins this bar covers
        bar_min = row['low']
        bar_max = row['high']

        for j in range(len(price_levels) - 1):
            level_min = price_levels[j]
            level_max = price_levels[j + 1]

            # Check overlap
            if bar_max >= level_min and bar_min <= level_max:
                # Calculate overlap percentage
                overlap_min = max(bar_min, level_min)
                overlap_max = min(bar_max, level_max)
                overlap_pct = (overlap_max - overlap_min) / (bar_max - bar_min) if bar_max > bar_min else 1.0

                # Distribute volume
                vol_contrib = row['volume'] * overlap_pct

                if is_up:
                    volume_up[j] += vol_contrib
                else:
                    volume_down[j] += vol_contrib

    # Use midpoint of bins for display
    price_levels_mid = (price_levels[:-1] + price_levels[1:]) / 2

    return price_levels_mid, volume_up, volume_down


def create_raindrop_chart(
    df: pd.DataFrame,
    title: Optional[str] = None,
    show_volume_profile: bool = True,
    profile_width: float = 0.3
) -> 'go.Figure':
    """
    Create raindrop chart with volume profile.

    Args:
        df: DataFrame with OHLCV data (must have index as datetime)
        title: Chart title
        show_volume_profile: Show volume profile bars
        profile_width: Width of profile bars (as fraction of chart)

    Returns:
        Plotly Figure object

    Examples:
        >>> fig = create_raindrop_chart(df, title='AAPL Raindrop Chart')
        >>> fig.show()

    Note:
        Requires plotly to be installed: pip install plotly
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for raindrop charts. Install: pip install plotly")

    # Calculate volume profile
    price_levels, vol_up, vol_down = calculate_volume_profile(df, price_bins=50)

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[1-profile_width, profile_width] if show_volume_profile else [1],
        shared_yaxes=True,
        horizontal_spacing=0.01
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )

    # Add volume profile if requested
    if show_volume_profile:
        # Volume up (green)
        fig.add_trace(
            go.Bar(
                x=vol_up,
                y=price_levels,
                orientation='h',
                name='Buy Volume',
                marker_color='rgba(0, 255, 0, 0.5)',
                showlegend=True
            ),
            row=1, col=2
        )

        # Volume down (red)
        fig.add_trace(
            go.Bar(
                x=-vol_down,  # Negative for left side
                y=price_levels,
                orientation='h',
                name='Sell Volume',
                marker_color='rgba(255, 0, 0, 0.5)',
                showlegend=True
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title=title or 'Raindrop Chart with Volume Profile',
        yaxis_title='Price',
        xaxis_title='Date',
        hovermode='x unified',
        height=600
    )

    # Hide x-axis for volume profile
    if show_volume_profile:
        fig.update_xaxes(showticklabels=False, row=1, col=2)

    return fig


def create_simple_raindrop(
    df: pd.DataFrame,
    symbol: str = 'Asset'
) -> 'go.Figure':
    """
    Create simple raindrop chart without profile.

    Args:
        df: DataFrame with OHLCV data
        symbol: Asset symbol for title

    Returns:
        Plotly Figure

    Examples:
        >>> fig = create_simple_raindrop(df, 'AAPL')
        >>> fig.write_html('raindrop.html')
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required. Install: pip install plotly")

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )
    ])

    fig.update_layout(
        title=f'{symbol} - Raindrop Chart',
        yaxis_title='Price',
        xaxis_title='Date',
        height=500
    )

    return fig


def is_plotly_available() -> bool:
    """Check if Plotly is available."""
    return PLOTLY_AVAILABLE


__all__ = [
    'create_raindrop_chart',
    'create_simple_raindrop',
    'calculate_volume_profile',
    'is_plotly_available',
]

__version__ = '2.0.0'
