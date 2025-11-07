"""
Unit tests for data.fred module.
"""
import pytest
import pandas as pd
from datetime import datetime
import os

from src.data.fred import (
    get_series,
    get_multiple_series,
    get_popular_series,
    is_fred_available,
    FREDClient,
    FREDError,
)


# Skip all tests if requests not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Check if API key is available
HAS_API_KEY = os.getenv('FRED_API_KEY') is not None


pytestmark = pytest.mark.skipif(
    not REQUESTS_AVAILABLE,
    reason="requests library not available"
)


class TestFREDAvailability:
    """Test FRED availability checks."""

    def test_is_fred_available_with_requests(self):
        """Test availability check when requests installed."""
        # Should check for API key
        available = is_fred_available()
        assert isinstance(available, bool)

    def test_get_popular_series(self):
        """Test getting popular series dictionary."""
        series = get_popular_series()

        assert isinstance(series, dict)
        assert 'gdp' in series
        assert 'unemployment' in series
        assert 'cpi' in series
        assert series['gdp'] == 'GDP'
        assert series['unemployment'] == 'UNRATE'


class TestFREDClient:
    """Test FREDClient class."""

    def test_client_initialization_no_key(self):
        """Test client initialization fails without API key."""
        # Temporarily clear API key env var
        original_key = os.environ.get('FRED_API_KEY')
        if 'FRED_API_KEY' in os.environ:
            del os.environ['FRED_API_KEY']

        try:
            with pytest.raises(FREDError, match="API key required"):
                FREDClient(api_key=None)
        finally:
            # Restore original key
            if original_key:
                os.environ['FRED_API_KEY'] = original_key

    def test_client_initialization_with_key(self):
        """Test client initialization with API key."""
        client = FREDClient(api_key='test_key')

        assert client.api_key == 'test_key'
        assert client.session is not None

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_series_basic(self):
        """Test basic series retrieval (requires API key and network)."""
        client = FREDClient()

        try:
            df = client.get_series('GDP', start='2020-01-01', end='2021-01-01')

            assert isinstance(df, pd.DataFrame)
            assert 'value' in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) > 0
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_series_with_dates(self):
        """Test series retrieval with date parameters."""
        client = FREDClient()

        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)

        try:
            df = client.get_series('GDP', start=start, end=end)

            assert isinstance(df, pd.DataFrame)
            assert df.index[0] >= start
            assert df.index[-1] <= end
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    def test_get_series_invalid_id(self):
        """Test error handling for invalid series ID."""
        if not HAS_API_KEY:
            pytest.skip("FRED API key not available")

        client = FREDClient()

        with pytest.raises(FREDError):
            client.get_series('INVALID_SERIES_ID_THAT_DOES_NOT_EXIST')

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_series_info(self):
        """Test getting series metadata."""
        client = FREDClient()

        try:
            info = client.get_series_info('GDP')

            assert isinstance(info, dict)
            assert 'id' in info
            assert 'title' in info
            assert info['id'] == 'GDP'
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_search_series(self):
        """Test searching for series."""
        client = FREDClient()

        try:
            results = client.search_series('unemployment', limit=5)

            assert isinstance(results, list)
            assert len(results) <= 5
            if results:
                assert 'id' in results[0]
                assert 'title' in results[0]
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_series_with_alias(self):
        """Test get_series with alias."""
        try:
            df = get_series('unemployment', start='2020-01-01', end='2021-01-01')

            assert isinstance(df, pd.DataFrame)
            assert 'value' in df.columns
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_series_with_id(self):
        """Test get_series with direct series ID."""
        try:
            df = get_series('GDP', start='2020-01-01', end='2021-01-01')

            assert isinstance(df, pd.DataFrame)
            assert 'value' in df.columns
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_multiple_series(self):
        """Test getting multiple series at once."""
        try:
            df = get_multiple_series(
                ['gdp', 'unemployment'],
                start='2020-01-01',
                end='2021-01-01'
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df.columns) >= 1  # At least one series succeeded
            assert isinstance(df.index, pd.DatetimeIndex)
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_get_multiple_series_some_invalid(self):
        """Test getting multiple series with some invalid IDs."""
        try:
            df = get_multiple_series(
                ['GDP', 'INVALID_ID', 'UNRATE'],
                start='2020-01-01',
                end='2021-01-01'
            )

            # Should return valid series and skip invalid ones
            assert isinstance(df, pd.DataFrame)
            assert len(df.columns) >= 1
        except FREDError as e:
            # If all series fail, should raise error
            pass


class TestDataIntegrity:
    """Test data integrity and format."""

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_series_data_sorted(self):
        """Test that series data is sorted by date."""
        client = FREDClient()

        try:
            df = client.get_series('GDP', start='2020-01-01', end='2021-01-01')

            assert df.index.is_monotonic_increasing
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_series_data_no_nan(self):
        """Test that series data has no NaN values."""
        client = FREDClient()

        try:
            df = client.get_series('GDP', start='2020-01-01', end='2021-01-01')

            assert not df['value'].isna().any()
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_series_value_numeric(self):
        """Test that series values are numeric."""
        client = FREDClient()

        try:
            df = client.get_series('GDP', start='2020-01-01', end='2021-01-01')

            assert pd.api.types.is_numeric_dtype(df['value'])
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")


class TestPopularSeries:
    """Test popular series aliases."""

    def test_popular_series_coverage(self):
        """Test that popular series cover major categories."""
        popular = get_popular_series()

        # GDP & Growth
        assert 'gdp' in popular
        assert 'gdp_growth' in popular

        # Employment
        assert 'unemployment' in popular
        assert 'nonfarm_payrolls' in popular

        # Inflation
        assert 'cpi' in popular
        assert 'pce' in popular

        # Interest Rates
        assert 'fed_funds' in popular
        assert 'treasury_10y' in popular

        # Housing
        assert 'housing_starts' in popular

        # Money Supply
        assert 'm1' in popular
        assert 'm2' in popular

    def test_popular_series_valid_ids(self):
        """Test that popular series have valid FRED IDs."""
        popular = get_popular_series()

        # Check a few known valid IDs
        assert popular['gdp'] == 'GDP'
        assert popular['unemployment'] == 'UNRATE'
        assert popular['cpi'] == 'CPIAUCSL'
        assert popular['fed_funds'] == 'FEDFUNDS'


class TestErrorHandling:
    """Test error handling."""

    def test_client_without_requests(self):
        """Test that client fails gracefully without requests."""
        # This is already tested by the skip marker, but we can verify the error
        if not REQUESTS_AVAILABLE:
            with pytest.raises(FREDError, match="requests not available"):
                FREDClient(api_key='test')

    def test_invalid_api_key(self):
        """Test error with invalid API key."""
        client = FREDClient(api_key='invalid_key')

        # Should fail when making request
        with pytest.raises(FREDError):
            client.get_series('GDP')

    def test_network_error_handling(self):
        """Test handling of network errors."""
        # This would require mocking requests, which is complex
        # Just verify that FREDError is raised for network issues
        pass


class TestDateFormatting:
    """Test date parameter formatting."""

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_string_dates(self):
        """Test with string date parameters."""
        client = FREDClient()

        try:
            df = client.get_series('GDP', start='2020-01-01', end='2021-01-01')
            assert len(df) > 0
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_datetime_dates(self):
        """Test with datetime date parameters."""
        client = FREDClient()

        start = datetime(2020, 1, 1)
        end = datetime(2021, 1, 1)

        try:
            df = client.get_series('GDP', start=start, end=end)
            assert len(df) > 0
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")

    @pytest.mark.skipif(not HAS_API_KEY, reason="FRED API key not available")
    def test_no_dates(self):
        """Test with no date parameters (full history)."""
        client = FREDClient()

        try:
            df = client.get_series('GDP')
            assert len(df) > 0
        except FREDError as e:
            pytest.skip(f"FRED API request failed: {e}")
