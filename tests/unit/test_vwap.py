"""
Unit tests for vwap module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from src.vwap import (
    AnchoredVWAP,
    calculate_vwap,
    calculate_vwap_from_timestamp,
    calculate_multiple_vwaps,
    session_anchor,
    week_anchor,
    month_anchor,
    swing_high_anchor,
    swing_low_anchor,
    timestamp_anchor,
    custom_anchor,
    get_anchor,
    AnchorError,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    return pd.DataFrame({
        'open': [100 + i for i in range(20)],
        'high': [105 + i for i in range(20)],
        'low': [98 + i for i in range(20)],
        'close': [102 + i for i in range(20)],
        'volume': [1000 + i*100 for i in range(20)]
    }, index=dates)


@pytest.fixture
def intraday_data():
    """Create intraday data for session anchor testing."""
    dates = pd.date_range('2024-01-01 09:30', periods=50, freq='1H')
    return pd.DataFrame({
        'open': [100 + i*0.5 for i in range(50)],
        'high': [102 + i*0.5 for i in range(50)],
        'low': [99 + i*0.5 for i in range(50)],
        'close': [101 + i*0.5 for i in range(50)],
        'volume': [1000 + i*50 for i in range(50)]
    }, index=dates)


class TestAnchoredVWAP:
    """Test AnchoredVWAP class."""

    def test_initialization(self):
        """Test VWAP calculator initialization."""
        vwap_calc = AnchoredVWAP()

        assert vwap_calc.include_bands is True
        assert vwap_calc.num_std == 1.0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        vwap_calc = AnchoredVWAP(include_bands=False, num_std=2.0)

        assert vwap_calc.include_bands is False
        assert vwap_calc.num_std == 2.0

    def test_calculate_session_anchor(self, sample_ohlcv):
        """Test VWAP calculation with session anchor."""
        vwap_calc = AnchoredVWAP()
        result = vwap_calc.calculate(sample_ohlcv, anchor_type='session')

        assert isinstance(result, pd.DataFrame)
        assert 'vwap' in result.columns
        assert len(result) == len(sample_ohlcv)
        assert not result['vwap'].isna().any()

    def test_calculate_with_bands(self, sample_ohlcv):
        """Test VWAP calculation includes bands."""
        vwap_calc = AnchoredVWAP(include_bands=True)
        result = vwap_calc.calculate(sample_ohlcv, anchor_type='session')

        assert 'vwap' in result.columns
        assert 'vwap_upper' in result.columns
        assert 'vwap_lower' in result.columns

        # Upper > VWAP > Lower
        assert (result['vwap_upper'] >= result['vwap']).all()
        assert (result['vwap'] >= result['vwap_lower']).all()

    def test_calculate_without_bands(self, sample_ohlcv):
        """Test VWAP calculation without bands."""
        vwap_calc = AnchoredVWAP(include_bands=False)
        result = vwap_calc.calculate(sample_ohlcv, anchor_type='session')

        assert 'vwap' in result.columns
        assert 'vwap_upper' not in result.columns
        assert 'vwap_lower' not in result.columns

    def test_calculate_from_timestamp(self, sample_ohlcv):
        """Test VWAP anchored to timestamp."""
        vwap_calc = AnchoredVWAP()
        result = vwap_calc.calculate(
            sample_ohlcv,
            anchor_timestamp='2024-01-05'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'vwap' in result.columns

    def test_calculate_missing_volume(self):
        """Test error when volume column missing."""
        df = pd.DataFrame({
            'close': [100, 101, 102]
        })

        vwap_calc = AnchoredVWAP()

        with pytest.raises(ValueError, match="volume"):
            vwap_calc.calculate(df, anchor_type='session')

    def test_calculate_custom_price_column(self, sample_ohlcv):
        """Test using custom price column."""
        vwap_calc = AnchoredVWAP()
        result = vwap_calc.calculate(
            sample_ohlcv,
            anchor_type='session',
            price_col='open'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'vwap' in result.columns


class TestAnchorFunctions:
    """Test anchor point detection functions."""

    def test_session_anchor(self, sample_ohlcv):
        """Test session anchor detection."""
        anchors = session_anchor(sample_ohlcv)

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool
        # First bar should be anchor
        assert anchors.iloc[0] == True

    def test_week_anchor(self, sample_ohlcv):
        """Test week anchor detection."""
        anchors = week_anchor(sample_ohlcv)

        assert isinstance(anchors, pd.Series)
        # dtype can be bool or BooleanDtype
        assert anchors.dtype in [bool, pd.BooleanDtype()]
        # At least one anchor should exist
        assert anchors.sum() > 0

    def test_month_anchor(self, sample_ohlcv):
        """Test month anchor detection."""
        anchors = month_anchor(sample_ohlcv)

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool
        # First bar should be anchor (start of month)
        assert anchors.iloc[0] == True

    def test_swing_high_anchor(self, sample_ohlcv):
        """Test swing high anchor detection."""
        anchors = swing_high_anchor(sample_ohlcv, lookback=5, threshold=0.01)

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool
        # May or may not find swings in this data

    def test_swing_low_anchor(self, sample_ohlcv):
        """Test swing low anchor detection."""
        anchors = swing_low_anchor(sample_ohlcv, lookback=5, threshold=0.01)

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool

    def test_timestamp_anchor(self, sample_ohlcv):
        """Test timestamp anchor."""
        anchors = timestamp_anchor(sample_ohlcv, '2024-01-10')

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool
        # Should have exactly one anchor
        assert anchors.sum() == 1

    def test_timestamp_anchor_exact_match(self, sample_ohlcv):
        """Test timestamp anchor with exact timestamp."""
        timestamp = sample_ohlcv.index[5]
        anchors = timestamp_anchor(sample_ohlcv, timestamp)

        assert anchors[timestamp] == True

    def test_custom_anchor(self, sample_ohlcv):
        """Test custom anchor function."""
        def every_5th_bar(df):
            result = pd.Series(False, index=df.index)
            result[::5] = True
            return result

        anchors = custom_anchor(sample_ohlcv, every_5th_bar)

        assert isinstance(anchors, pd.Series)
        assert anchors.dtype == bool
        # Every 5th bar should be anchor
        assert anchors.iloc[0] == True
        assert anchors.iloc[5] == True
        assert anchors.iloc[10] == True

    def test_get_anchor(self, sample_ohlcv):
        """Test get_anchor utility."""
        anchors = get_anchor(sample_ohlcv, 'session')

        assert isinstance(anchors, pd.Series)

    def test_get_anchor_invalid_type(self, sample_ohlcv):
        """Test error with invalid anchor type."""
        with pytest.raises(AnchorError, match="Unknown anchor type"):
            get_anchor(sample_ohlcv, 'invalid_type')


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_calculate_vwap(self, sample_ohlcv):
        """Test calculate_vwap convenience function."""
        result = calculate_vwap(sample_ohlcv, anchor_type='session')

        assert isinstance(result, pd.DataFrame)
        assert 'vwap' in result.columns

    def test_calculate_vwap_with_params(self, sample_ohlcv):
        """Test calculate_vwap with custom parameters."""
        result = calculate_vwap(
            sample_ohlcv,
            anchor_type='session',
            include_bands=True,
            num_std=2.0
        )

        assert 'vwap' in result.columns
        assert 'vwap_upper' in result.columns

    def test_calculate_vwap_from_timestamp(self, sample_ohlcv):
        """Test calculate_vwap_from_timestamp function."""
        result = calculate_vwap_from_timestamp(sample_ohlcv, '2024-01-05')

        assert isinstance(result, pd.DataFrame)
        assert 'vwap' in result.columns

    def test_calculate_multiple_vwaps(self, sample_ohlcv):
        """Test calculating multiple VWAPs."""
        result = calculate_multiple_vwaps(
            sample_ohlcv,
            ['session', 'week', 'month']
        )

        assert isinstance(result, pd.DataFrame)
        assert 'vwap_session' in result.columns
        assert 'vwap_week' in result.columns
        assert 'vwap_month' in result.columns

    def test_calculate_multiple_vwaps_with_bands(self, sample_ohlcv):
        """Test calculating multiple VWAPs with bands."""
        result = calculate_multiple_vwaps(
            sample_ohlcv,
            ['session', 'week'],
            include_bands=True
        )

        assert 'vwap_session' in result.columns
        assert 'vwap_session_upper' in result.columns
        assert 'vwap_session_lower' in result.columns
        assert 'vwap_week' in result.columns
        assert 'vwap_week_upper' in result.columns


class TestVWAPValues:
    """Test VWAP calculation correctness."""

    def test_vwap_calculation(self):
        """Test VWAP calculation with known values."""
        df = pd.DataFrame({
            'close': [100, 110, 105],
            'volume': [1000, 2000, 1500]
        }, index=pd.date_range('2024-01-01 09:30', periods=3, freq='H'))

        # Use timestamp anchor at first bar so VWAP cumulates across all bars
        result = calculate_vwap_from_timestamp(df, df.index[0], include_bands=False)

        # VWAP should be volume-weighted average
        # Bar 1: (100*1000) / 1000 = 100
        # Bar 2: (100*1000 + 110*2000) / (1000+2000) = 320000/3000 = 106.67
        # Bar 3: (100*1000 + 110*2000 + 105*1500) / (1000+2000+1500) = 477500/4500 = 106.11

        assert abs(result['vwap'].iloc[0] - 100) < 0.01
        assert abs(result['vwap'].iloc[1] - 106.67) < 0.01
        assert abs(result['vwap'].iloc[2] - 106.11) < 0.01

    def test_vwap_multiple_anchors(self):
        """Test VWAP resets at each anchor."""
        df = pd.DataFrame({
            'close': [100, 110, 105, 120],
            'volume': [1000, 2000, 1500, 1000]
        }, index=pd.date_range('2024-01-01', periods=4, freq='D'))

        # Create custom anchor at bars 0 and 2
        def custom_anchors(df):
            result = pd.Series(False, index=df.index)
            result.iloc[0] = True
            result.iloc[2] = True
            return result

        vwap_calc = AnchoredVWAP(include_bands=False)
        result = vwap_calc.calculate(df, anchor_func=custom_anchors)

        # Period 1 (bars 0-1): same as before
        # Period 2 (bars 2-3): should reset
        # Bar 2: 105
        # Bar 3: (105*1500 + 120*1000) / (1500+1000) = 277500/2500 = 111

        assert abs(result['vwap'].iloc[2] - 105) < 0.01
        assert abs(result['vwap'].iloc[3] - 111) < 0.01


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['close', 'volume'])

        vwap_calc = AnchoredVWAP()

        with pytest.raises(Exception):
            vwap_calc.calculate(df, anchor_type='session')

    def test_single_bar(self):
        """Test with single bar."""
        df = pd.DataFrame({
            'close': [100],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))

        result = calculate_vwap(df, anchor_type='session', include_bands=False)

        assert len(result) == 1
        assert abs(result['vwap'].iloc[0] - 100) < 0.01

    def test_zero_volume(self):
        """Test handling of zero volume."""
        df = pd.DataFrame({
            'close': [100, 110, 105],
            'volume': [1000, 0, 1500]
        }, index=pd.date_range('2024-01-01', periods=3, freq='D'))

        result = calculate_vwap(df, anchor_type='session', include_bands=False)

        # Should handle zero volume gracefully
        assert not result['vwap'].isna().any()

    def test_anchor_without_datetime_index(self):
        """Test error when DataFrame lacks DatetimeIndex."""
        df = pd.DataFrame({
            'close': [100, 110, 105],
            'volume': [1000, 2000, 1500]
        })

        with pytest.raises(AnchorError):
            session_anchor(df)
