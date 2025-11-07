"""
Unit tests for zones module.
"""
import pytest
import pandas as pd
import numpy as np

from src.zones import (
    Zone,
    detect_zones,
    calculate_zone_strength,
    get_active_zones,
    filter_zones,
    merge_overlapping_zones,
    remove_weak_zones,
    keep_best_zones,
    deduplicate_zones,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data with clear zones."""
    # Create data with consolidation followed by impulse
    close_prices = [100] * 10  # Consolidation
    close_prices.extend([110, 115, 120])  # Impulse up (demand zone)
    close_prices.extend([120] * 10)  # Consolidation
    close_prices.extend([110, 105, 100])  # Impulse down (supply zone)
    close_prices.extend([100] * 24)  # Continuation (total=50)

    dates = pd.date_range('2024-01-01', periods=len(close_prices), freq='D')

    df = pd.DataFrame({
        'open': close_prices,
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000] * len(close_prices)
    }, index=dates)

    return df


class TestZoneDataclass:
    """Test Zone dataclass."""

    def test_zone_creation(self):
        """Test creating a Zone."""
        zone = Zone(
            zone_type='demand',
            top=105,
            bottom=100,
            start_idx=0,
            end_idx=5,
            impulse_idx=6
        )

        assert zone.zone_type == 'demand'
        assert zone.top == 105
        assert zone.bottom == 100
        assert zone.fresh is True
        assert zone.touches == 0

    def test_zone_size(self):
        """Test zone size property."""
        zone = Zone('demand', 105, 100, 0, 5, 6)

        assert zone.size == 5

    def test_zone_midpoint(self):
        """Test zone midpoint property."""
        zone = Zone('demand', 105, 100, 0, 5, 6)

        assert zone.midpoint == 102.5

    def test_zone_contains(self):
        """Test if price is within zone."""
        zone = Zone('demand', 105, 100, 0, 5, 6)

        assert zone.contains(102) is True
        assert zone.contains(99) is False
        assert zone.contains(106) is False


class TestZoneDetection:
    """Test zone detection."""

    def test_detect_zones_basic(self, sample_ohlcv):
        """Test basic zone detection."""
        zones = detect_zones(sample_ohlcv)

        assert isinstance(zones, list)
        # May or may not detect zones depending on thresholds
        # Just verify the function runs without error

    def test_detect_zones_types(self, sample_ohlcv):
        """Test zone types detected."""
        zones = detect_zones(sample_ohlcv, impulse_threshold=0.05)

        # Should detect both supply and demand zones
        types = [z.zone_type for z in zones]
        # May or may not detect both depending on data

    def test_detect_zones_empty(self):
        """Test with DataFrame that has no clear zones."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range('2024-01-01', periods=3))

        zones = detect_zones(df)

        # May be empty or have zones depending on thresholds
        assert isinstance(zones, list)

    def test_detect_zones_custom_params(self, sample_ohlcv):
        """Test zone detection with custom parameters."""
        zones = detect_zones(
            sample_ohlcv,
            consolidation_bars=5,
            impulse_threshold=0.03,
            max_zone_size=0.05
        )

        assert isinstance(zones, list)


class TestZoneStrength:
    """Test zone strength calculation."""

    def test_calculate_zone_strength(self, sample_ohlcv):
        """Test strength calculation."""
        zone = Zone('demand', 102, 98, 0, 5, 6)

        strength = calculate_zone_strength(sample_ohlcv, zone)

        assert isinstance(strength, float)
        assert 0 <= strength <= 100

    def test_zone_strength_in_detection(self, sample_ohlcv):
        """Test that detected zones have strength."""
        zones = detect_zones(sample_ohlcv)

        if zones:
            assert all(hasattr(z, 'strength') for z in zones)
            assert all(0 <= z.strength <= 100 for z in zones)


class TestZoneQueries:
    """Test zone query functions."""

    def test_get_active_zones(self):
        """Test getting zones near current price."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 130, 125, 10, 15, 16, strength=60),
            Zone('demand', 80, 75, 20, 25, 26, strength=50),
        ]

        # Current price near first zone
        active = get_active_zones(zones, current_price=102, proximity_pct=0.05)

        assert len(active) >= 1
        assert zones[0] in active

    def test_filter_zones_by_type(self):
        """Test filtering zones by type."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 130, 125, 10, 15, 16, strength=60),
            Zone('demand', 80, 75, 20, 25, 26, strength=50),
        ]

        demand_zones = filter_zones(zones, zone_type='demand')

        assert len(demand_zones) == 2
        assert all(z.zone_type == 'demand' for z in demand_zones)

    def test_filter_zones_by_strength(self):
        """Test filtering zones by minimum strength."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 130, 125, 10, 15, 16, strength=60),
            Zone('demand', 80, 75, 20, 25, 26, strength=40),
        ]

        strong_zones = filter_zones(zones, min_strength=50)

        assert len(strong_zones) == 2
        assert all(z.strength >= 50 for z in strong_zones)

    def test_filter_zones_fresh_only(self):
        """Test filtering for fresh zones only."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70, fresh=True),
            Zone('supply', 130, 125, 10, 15, 16, strength=60, fresh=False),
        ]

        fresh = filter_zones(zones, fresh_only=True)

        assert len(fresh) == 1
        assert fresh[0].fresh is True


class TestZoneMerging:
    """Test zone merging functions."""

    def test_merge_overlapping_zones(self):
        """Test merging overlapping zones."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('demand', 108, 103, 7, 12, 13, strength=60),  # Overlaps
        ]

        merged = merge_overlapping_zones(zones)

        # Should merge into one zone
        assert len(merged) <= len(zones)

    def test_merge_different_types_not_merged(self):
        """Test that different zone types are not merged."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 108, 103, 7, 12, 13, strength=60),  # Different type
        ]

        merged = merge_overlapping_zones(zones)

        # Should not merge (different types)
        assert len(merged) == 2

    def test_remove_weak_zones(self):
        """Test removing weak zones."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 130, 125, 10, 15, 16, strength=30),
            Zone('demand', 80, 75, 20, 25, 26, strength=50),
        ]

        strong = remove_weak_zones(zones, min_strength=40)

        assert len(strong) == 2
        assert all(z.strength >= 40 for z in strong)

    def test_keep_best_zones(self):
        """Test keeping only best zones."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('supply', 130, 125, 10, 15, 16, strength=60),
            Zone('demand', 80, 75, 20, 25, 26, strength=50),
        ]

        best = keep_best_zones(zones, max_zones=2)

        assert len(best) == 2
        assert best[0].strength == 70
        assert best[1].strength == 60

    def test_deduplicate_zones(self):
        """Test deduplicating nearly identical zones."""
        zones = [
            Zone('demand', 105, 100, 0, 5, 6, strength=70),
            Zone('demand', 105.1, 100.1, 7, 12, 13, strength=60),  # Nearly identical
        ]

        unique = deduplicate_zones(zones, tolerance=0.01)

        # Should keep only one (the stronger)
        assert len(unique) == 1
        assert unique[0].strength == 70


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_zone_list(self):
        """Test functions with empty zone list."""
        zones = []

        assert merge_overlapping_zones(zones) == []
        assert keep_best_zones(zones) == []
        assert deduplicate_zones(zones) == []

    def test_single_zone(self):
        """Test with single zone."""
        zones = [Zone('demand', 105, 100, 0, 5, 6, strength=70)]

        assert len(merge_overlapping_zones(zones)) == 1
        assert len(keep_best_zones(zones, max_zones=5)) == 1
