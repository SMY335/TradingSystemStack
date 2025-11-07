"""
Supply and Demand zone detection for TradingSystemStack.

Identifies price zones where supply/demand imbalances create trading opportunities.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Supply or Demand zone.

    Attributes:
        zone_type: 'supply' or 'demand'
        top: Upper price boundary
        bottom: Lower price boundary
        start_idx: Start index in dataframe
        end_idx: End index (consolidation end)
        impulse_idx: Impulse move index
        strength: Zone strength (0-100)
        touches: Number of times price touched zone
        fresh: Whether zone is untouched since creation
    """
    zone_type: str
    top: float
    bottom: float
    start_idx: int
    end_idx: int
    impulse_idx: int
    strength: float = 0.0
    touches: int = 0
    fresh: bool = True

    @property
    def size(self) -> float:
        """Zone size in price units."""
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        """Zone midpoint."""
        return (self.top + self.bottom) / 2

    def contains(self, price: float) -> bool:
        """Check if price is within zone."""
        return self.bottom <= price <= self.top

    def __repr__(self) -> str:
        return (f"Zone({self.zone_type}, {self.bottom:.2f}-{self.top:.2f}, "
                f"strength={self.strength:.1f}, touches={self.touches}, fresh={self.fresh})")


def detect_zones(
    df: pd.DataFrame,
    consolidation_bars: int = 3,
    impulse_threshold: float = 0.02,
    max_zone_size: float = 0.03
) -> List[Zone]:
    """Detect supply and demand zones.

    Args:
        df: DataFrame with OHLC data
        consolidation_bars: Minimum consolidation bars
        impulse_threshold: Minimum % move for impulse (0.02 = 2%)
        max_zone_size: Maximum zone size as % (0.03 = 3%)

    Returns:
        List of detected zones

    Examples:
        >>> zones = detect_zones(df)
        >>> zones = detect_zones(df, consolidation_bars=5, impulse_threshold=0.03)
    """
    zones = []

    # Find consolidation periods followed by impulse moves
    for i in range(consolidation_bars, len(df) - 1):
        # Check for consolidation (low volatility)
        consolidation_start = i - consolidation_bars
        consolidation_range = (
            df['high'].iloc[consolidation_start:i].max() -
            df['low'].iloc[consolidation_start:i].min()
        )

        avg_price = df['close'].iloc[consolidation_start:i].mean()
        consolidation_pct = consolidation_range / avg_price

        # Consolidation should be tight
        if consolidation_pct > max_zone_size:
            continue

        # Check for impulse move after consolidation
        impulse_move = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]

        # Demand zone: consolidation followed by strong up move
        if impulse_move > impulse_threshold:
            zone = Zone(
                zone_type='demand',
                top=df['high'].iloc[consolidation_start:i].max(),
                bottom=df['low'].iloc[consolidation_start:i].min(),
                start_idx=consolidation_start,
                end_idx=i,
                impulse_idx=i+1
            )
            zone.strength = calculate_zone_strength(df, zone)
            zones.append(zone)

        # Supply zone: consolidation followed by strong down move
        elif impulse_move < -impulse_threshold:
            zone = Zone(
                zone_type='supply',
                top=df['high'].iloc[consolidation_start:i].max(),
                bottom=df['low'].iloc[consolidation_start:i].min(),
                start_idx=consolidation_start,
                end_idx=i,
                impulse_idx=i+1
            )
            zone.strength = calculate_zone_strength(df, zone)
            zones.append(zone)

    # Update zones with touch counts
    zones = update_zone_touches(df, zones)

    return zones


def calculate_zone_strength(df: pd.DataFrame, zone: Zone) -> float:
    """Calculate zone strength based on multiple factors.

    Args:
        df: DataFrame with OHLC data
        zone: Zone to evaluate

    Returns:
        Strength score (0-100)
    """
    score = 0.0

    # Factor 1: Impulse strength (0-40 points)
    if zone.impulse_idx < len(df):
        impulse_move = abs(
            (df['close'].iloc[zone.impulse_idx] - df['close'].iloc[zone.end_idx]) /
            df['close'].iloc[zone.end_idx]
        )
        impulse_score = min(impulse_move * 1000, 40)  # Cap at 40
        score += impulse_score

    # Factor 2: Zone tightness (0-30 points)
    zone_size_pct = zone.size / zone.midpoint
    tightness_score = max(30 - (zone_size_pct * 1000), 0)
    score += tightness_score

    # Factor 3: Volume at formation (0-30 points)
    if 'volume' in df.columns and zone.end_idx < len(df):
        zone_volume = df['volume'].iloc[zone.start_idx:zone.end_idx+1].mean()
        avg_volume = df['volume'].mean()
        if avg_volume > 0:
            volume_ratio = zone_volume / avg_volume
            volume_score = min(volume_ratio * 15, 30)
            score += volume_score

    return min(score, 100)


def update_zone_touches(df: pd.DataFrame, zones: List[Zone]) -> List[Zone]:
    """Update zone touch counts and freshness.

    Args:
        df: DataFrame with OHLC data
        zones: List of zones to update

    Returns:
        Updated zones
    """
    for zone in zones:
        # Count touches after zone formation
        for i in range(zone.impulse_idx + 1, len(df)):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Check if price touched zone
            if zone.contains(high) or zone.contains(low) or \
               (low < zone.bottom and high > zone.top):
                zone.touches += 1

                # Zone is no longer fresh after first touch
                if zone.touches == 1:
                    zone.fresh = False

    return zones


def get_active_zones(
    zones: List[Zone],
    current_price: float,
    proximity_pct: float = 0.05
) -> List[Zone]:
    """Get zones near current price.

    Args:
        zones: List of all zones
        current_price: Current price level
        proximity_pct: Proximity threshold (0.05 = 5%)

    Returns:
        List of nearby zones

    Examples:
        >>> active = get_active_zones(zones, current_price=100, proximity_pct=0.02)
    """
    threshold = current_price * proximity_pct

    active = []
    for zone in zones:
        # Distance from current price to zone
        if current_price > zone.top:
            distance = current_price - zone.top
        elif current_price < zone.bottom:
            distance = zone.bottom - current_price
        else:
            distance = 0  # Inside zone

        if distance <= threshold:
            active.append(zone)

    return active


def project_zones_forward(
    zones: List[Zone],
    df: pd.DataFrame,
    lookback: Optional[int] = None
) -> pd.DataFrame:
    """Project zones onto price chart.

    Args:
        zones: List of zones
        df: DataFrame to project onto
        lookback: Only include zones from last N bars

    Returns:
        DataFrame with zone levels

    Examples:
        >>> zone_df = project_zones_forward(zones, df)
    """
    result = pd.DataFrame(index=df.index)

    for i, zone in enumerate(zones):
        if lookback and zone.end_idx < len(df) - lookback:
            continue

        # Add zone boundaries as columns
        zone_name = f"{zone.zone_type}_{i}"
        result[f"{zone_name}_top"] = zone.top
        result[f"{zone_name}_bottom"] = zone.bottom

    return result


def filter_zones(
    zones: List[Zone],
    zone_type: Optional[str] = None,
    min_strength: float = 0.0,
    fresh_only: bool = False,
    max_touches: Optional[int] = None
) -> List[Zone]:
    """Filter zones by criteria.

    Args:
        zones: List of zones
        zone_type: 'supply' or 'demand' (None = both)
        min_strength: Minimum strength threshold
        fresh_only: Only untouched zones
        max_touches: Maximum touch count

    Returns:
        Filtered zones

    Examples:
        >>> demand_zones = filter_zones(zones, zone_type='demand', min_strength=50)
        >>> fresh_zones = filter_zones(zones, fresh_only=True)
    """
    filtered = zones

    if zone_type:
        filtered = [z for z in filtered if z.zone_type == zone_type]

    if min_strength > 0:
        filtered = [z for z in filtered if z.strength >= min_strength]

    if fresh_only:
        filtered = [z for z in filtered if z.fresh]

    if max_touches is not None:
        filtered = [z for z in filtered if z.touches <= max_touches]

    return filtered
