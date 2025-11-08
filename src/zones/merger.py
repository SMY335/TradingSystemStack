"""
Zone merging utilities for overlapping or nearby zones.
"""
from typing import List
import logging

from .supply_demand import Zone

logger = logging.getLogger(__name__)


def merge_overlapping_zones(zones: List[Zone], merge_threshold: float = 0.0) -> List[Zone]:
    """Merge overlapping or nearby zones of the same type.

    Args:
        zones: List of zones to merge
        merge_threshold: Distance threshold for merging (0 = only overlapping)

    Returns:
        List of merged zones

    Examples:
        >>> merged = merge_overlapping_zones(zones)
        >>> merged = merge_overlapping_zones(zones, merge_threshold=0.01)
    """
    if not zones:
        return []

    # Sort zones by type and bottom price
    sorted_zones = sorted(zones, key=lambda z: (z.zone_type, z.bottom))

    merged = []
    current_zone = sorted_zones[0]

    for next_zone in sorted_zones[1:]:
        # Only merge zones of same type
        if current_zone.zone_type != next_zone.zone_type:
            merged.append(current_zone)
            current_zone = next_zone
            continue

        # Calculate distance between zones
        distance = next_zone.bottom - current_zone.top

        # Merge if overlapping or within threshold
        if distance <= merge_threshold:
            # Merge zones
            current_zone = _merge_two_zones(current_zone, next_zone)
            logger.debug(f"Merged zones: {current_zone}")
        else:
            # No merge, save current and move to next
            merged.append(current_zone)
            current_zone = next_zone

    # Add last zone
    merged.append(current_zone)

    return merged


def _merge_two_zones(zone1: Zone, zone2: Zone) -> Zone:
    """Merge two zones into one.

    Args:
        zone1: First zone
        zone2: Second zone

    Returns:
        Merged zone
    """
    # Take widest boundaries
    merged_zone = Zone(
        zone_type=zone1.zone_type,
        top=max(zone1.top, zone2.top),
        bottom=min(zone1.bottom, zone2.bottom),
        start_idx=min(zone1.start_idx, zone2.start_idx),
        end_idx=max(zone1.end_idx, zone2.end_idx),
        impulse_idx=zone1.impulse_idx if zone1.strength > zone2.strength else zone2.impulse_idx,
        strength=max(zone1.strength, zone2.strength),
        touches=zone1.touches + zone2.touches,
        fresh=zone1.fresh and zone2.fresh
    )

    return merged_zone


def remove_weak_zones(zones: List[Zone], min_strength: float = 30.0) -> List[Zone]:
    """Remove zones below strength threshold.

    Args:
        zones: List of zones
        min_strength: Minimum strength to keep

    Returns:
        Filtered zones

    Examples:
        >>> strong_zones = remove_weak_zones(zones, min_strength=50)
    """
    return [z for z in zones if z.strength >= min_strength]


def keep_best_zones(zones: List[Zone], max_zones: int = 10) -> List[Zone]:
    """Keep only the strongest zones.

    Args:
        zones: List of zones
        max_zones: Maximum number to keep

    Returns:
        Top zones by strength

    Examples:
        >>> best = keep_best_zones(zones, max_zones=5)
    """
    # Sort by strength (descending)
    sorted_zones = sorted(zones, key=lambda z: z.strength, reverse=True)

    return sorted_zones[:max_zones]


def deduplicate_zones(zones: List[Zone], tolerance: float = 0.001) -> List[Zone]:
    """Remove duplicate zones (nearly identical boundaries).

    Args:
        zones: List of zones
        tolerance: Price tolerance for considering zones identical (0.001 = 0.1%)

    Returns:
        Deduplicated zones

    Examples:
        >>> unique = deduplicate_zones(zones)
    """
    if not zones:
        return []

    unique_zones = [zones[0]]

    for zone in zones[1:]:
        is_duplicate = False

        for existing in unique_zones:
            # Check if zones are nearly identical
            if zone.zone_type == existing.zone_type:
                top_diff = abs(zone.top - existing.top) / existing.top
                bottom_diff = abs(zone.bottom - existing.bottom) / existing.bottom

                if top_diff < tolerance and bottom_diff < tolerance:
                    is_duplicate = True
                    # Keep stronger zone
                    if zone.strength > existing.strength:
                        unique_zones.remove(existing)
                        unique_zones.append(zone)
                    break

        if not is_duplicate:
            unique_zones.append(zone)

    return unique_zones
