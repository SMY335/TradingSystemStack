"""
Supply and Demand zone detection for TradingSystemStack.

Identifies price zones where institutional supply/demand creates
trading opportunities.

Examples:
    >>> from src.zones import detect_zones, merge_overlapping_zones, Zone
    >>>
    >>> # Detect zones
    >>> zones = detect_zones(df, consolidation_bars=5, impulse_threshold=0.02)
    >>>
    >>> # Filter zones
    >>> demand_zones = filter_zones(zones, zone_type='demand', min_strength=50)
    >>> fresh_zones = filter_zones(zones, fresh_only=True)
    >>>
    >>> # Merge overlapping zones
    >>> merged = merge_overlapping_zones(zones)
    >>>
    >>> # Get zones near current price
    >>> active = get_active_zones(zones, current_price=100, proximity_pct=0.02)
"""

from .supply_demand import (
    Zone,
    detect_zones,
    calculate_zone_strength,
    update_zone_touches,
    get_active_zones,
    project_zones_forward,
    filter_zones,
)

from .merger import (
    merge_overlapping_zones,
    remove_weak_zones,
    keep_best_zones,
    deduplicate_zones,
)


__all__ = [
    # Core types
    'Zone',

    # Detection
    'detect_zones',
    'calculate_zone_strength',
    'update_zone_touches',

    # Zone queries
    'get_active_zones',
    'project_zones_forward',
    'filter_zones',

    # Merging
    'merge_overlapping_zones',
    'remove_weak_zones',
    'keep_best_zones',
    'deduplicate_zones',
]
