"""
Unit tests for data.normalizers module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.normalizers import (
    normalize_columns,
    validate_ohlcv,
    ensure_numeric_types,
    remove_invalid_rows,
    ensure_sorted,
    remove_duplicates,
    normalize_ohlcv,
    detect_data_issues,
    get_required_columns,
    get_column_aliases,
)


class TestNormalizeColumns:
    """Test normalize_columns function."""

    def test_normalize_capitalized_columns(self):
        """Test normalizing capitalized column names."""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 2000]
        })

        normalized = normalize_columns(df)

        assert 'open' in normalized.columns
        assert 'high' in normalized.columns
        assert 'low' in normalized.columns
        assert 'close' in normalized.columns
        assert 'volume' in normalized.columns

    def test_normalize_uppercase_columns(self):
        """Test normalizing uppercase column names."""
        df = pd.DataFrame({
            'OPEN': [100],
            'HIGH': [102],
            'LOW': [99],
            'CLOSE': [101],
            'VOLUME': [1000]
        })

        normalized = normalize_columns(df)

        assert list(normalized.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_normalize_already_lowercase(self):
        """Test that lowercase columns remain unchanged."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })

        normalized = normalize_columns(df)

        assert list(normalized.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_normalize_adj_close(self):
        """Test that Adj Close is normalized to close."""
        df = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Adj Close': [101],
            'Volume': [1000]
        })

        normalized = normalize_columns(df)

        assert 'close' in normalized.columns
        assert normalized['close'].iloc[0] == 101

    def test_keep_extra_columns(self):
        """Test keeping non-OHLCV columns."""
        df = pd.DataFrame({
            'Open': [100],
            'Close': [101],
            'CustomColumn': ['value']
        })

        normalized = normalize_columns(df, keep_extra=True)

        assert 'customcolumn' in normalized.columns
        assert normalized['customcolumn'].iloc[0] == 'value'

    def test_remove_extra_columns(self):
        """Test removing non-OHLCV columns."""
        df = pd.DataFrame({
            'Open': [100],
            'Close': [101],
            'CustomColumn': ['value']
        })

        normalized = normalize_columns(df, keep_extra=False)

        assert 'customcolumn' not in normalized.columns

    def test_custom_rename_map(self):
        """Test custom column rename mapping."""
        df = pd.DataFrame({
            'price_open': [100],
            'price_close': [101]
        })

        rename_map = {
            'open': ['price_open'],
            'close': ['price_close']
        }

        normalized = normalize_columns(df, rename_map=rename_map)

        assert 'open' in normalized.columns
        assert 'close' in normalized.columns


class TestValidateOHLCV:
    """Test validate_ohlcv function."""

    def test_validate_complete_ohlcv(self):
        """Test validation with all required columns."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })

        assert validate_ohlcv(df) is True

    def test_validate_missing_volume_required(self):
        """Test validation fails when volume missing and required."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101]
        })

        with pytest.raises(ValueError, match="Missing required"):
            validate_ohlcv(df, require_volume=True)

    def test_validate_missing_volume_optional(self):
        """Test validation passes when volume missing but not required."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101]
        })

        assert validate_ohlcv(df, require_volume=False) is True

    def test_validate_missing_ohlc(self):
        """Test validation fails with missing OHLC columns."""
        df = pd.DataFrame({
            'open': [100],
            'close': [101]
        })

        with pytest.raises(ValueError, match="Missing required"):
            validate_ohlcv(df)

    def test_validate_no_raise(self):
        """Test validation without raising exception."""
        df = pd.DataFrame({'open': [100]})

        result = validate_ohlcv(df, raise_on_missing=False)
        assert result is False


class TestEnsureNumericTypes:
    """Test ensure_numeric_types function."""

    def test_convert_string_to_numeric(self):
        """Test converting string columns to numeric."""
        df = pd.DataFrame({
            'open': ['100.0', '101.0'],
            'close': ['102.0', '103.0'],
            'volume': ['1000', '2000']
        })

        numeric_df = ensure_numeric_types(df)

        assert pd.api.types.is_numeric_dtype(numeric_df['open'])
        assert pd.api.types.is_numeric_dtype(numeric_df['close'])
        assert pd.api.types.is_numeric_dtype(numeric_df['volume'])

    def test_handle_invalid_values(self):
        """Test that invalid values become NaN."""
        df = pd.DataFrame({
            'open': ['100.0', 'invalid'],
            'close': ['102.0', '103.0']
        })

        numeric_df = ensure_numeric_types(df)

        assert pd.isna(numeric_df['open'].iloc[1])


class TestRemoveInvalidRows:
    """Test remove_invalid_rows function."""

    def test_remove_nan_ohlc(self):
        """Test removing rows with NaN in OHLC."""
        df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 2000, 3000]
        })

        clean = remove_invalid_rows(df, check_ohlc=True)

        assert len(clean) == 2
        assert clean['open'].iloc[0] == 100
        assert clean['open'].iloc[1] == 102

    def test_remove_zero_volume(self):
        """Test removing rows with zero volume."""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [101, 102, 103],
            'volume': [1000, 0, 3000]
        })

        clean = remove_invalid_rows(df, remove_zero_volume=True)

        assert len(clean) == 2
        assert clean['volume'].iloc[0] == 1000
        assert clean['volume'].iloc[1] == 3000

    def test_remove_nan_volume(self):
        """Test removing rows with NaN volume."""
        df = pd.DataFrame({
            'open': [100, 101],
            'close': [101, 102],
            'volume': [1000, np.nan]
        })

        clean = remove_invalid_rows(df, check_volume=True)

        assert len(clean) == 1


class TestEnsureSorted:
    """Test ensure_sorted function."""

    def test_sort_unsorted_index(self):
        """Test sorting unsorted DataFrame."""
        dates = pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-02'])
        df = pd.DataFrame({'close': [100, 101, 102]}, index=dates)

        sorted_df = ensure_sorted(df)

        assert sorted_df.index[0] == pd.Timestamp('2024-01-01')
        assert sorted_df.index[1] == pd.Timestamp('2024-01-02')
        assert sorted_df.index[2] == pd.Timestamp('2024-01-03')

    def test_already_sorted(self):
        """Test that already sorted DataFrame remains unchanged."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        df = pd.DataFrame({'close': [100, 101, 102]}, index=dates)

        sorted_df = ensure_sorted(df)

        assert sorted_df.index[0] == pd.Timestamp('2024-01-01')


class TestRemoveDuplicates:
    """Test remove_duplicates function."""

    def test_remove_duplicate_timestamps_last(self):
        """Test removing duplicates, keeping last."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        df = pd.DataFrame({'close': [100, 101, 102]}, index=dates)

        dedupe = remove_duplicates(df, keep='last')

        assert len(dedupe) == 2
        assert dedupe['close'].iloc[0] == 101  # Kept last duplicate

    def test_remove_duplicate_timestamps_first(self):
        """Test removing duplicates, keeping first."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        df = pd.DataFrame({'close': [100, 101, 102]}, index=dates)

        dedupe = remove_duplicates(df, keep='first')

        assert len(dedupe) == 2
        assert dedupe['close'].iloc[0] == 100  # Kept first duplicate


class TestNormalizeOHLCV:
    """Test normalize_ohlcv complete pipeline."""

    def test_complete_normalization(self):
        """Test complete normalization pipeline."""
        dates = pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-01', '2024-01-02'])
        df = pd.DataFrame({
            'Open': ['100', '101', '101.5', '102'],
            'High': ['102', '103', '103', '104'],
            'Low': ['99', '100', '100', '101'],
            'Close': ['101', '102', '102', '103'],
            'Volume': ['1000', '0', '2000', '3000']
        }, index=dates)

        normalized = normalize_ohlcv(
            df,
            validate=True,
            ensure_numeric=True,
            remove_invalid=True,
            remove_zero_volume=True,
            sort=True,
            deduplicate=True
        )

        # Check columns normalized
        assert 'open' in normalized.columns

        # Check numeric types
        assert normalized['open'].dtype == np.float64

        # Check zero volume removed
        assert 0 not in normalized['volume'].values

        # Check sorted
        assert normalized.index[0] == pd.Timestamp('2024-01-01')

        # Check duplicates removed
        assert len(normalized) == 3


class TestDetectDataIssues:
    """Test detect_data_issues function."""

    def test_detect_nan(self):
        """Test detecting NaN values."""
        df = pd.DataFrame({
            'open': [100, np.nan],
            'close': [101, 102]
        })

        issues = detect_data_issues(df)

        assert issues['has_nan'] is True
        assert issues['nan_count'] == 1
        assert 'open' in issues['nan_columns']

    def test_detect_zero_volume(self):
        """Test detecting zero volume."""
        df = pd.DataFrame({
            'close': [100, 101],
            'volume': [1000, 0]
        })

        issues = detect_data_issues(df)

        assert issues['zero_volume_count'] == 1

    def test_detect_negative_values(self):
        """Test detecting negative values."""
        df = pd.DataFrame({
            'open': [100, -50],
            'close': [101, 102]
        })

        issues = detect_data_issues(df)

        assert issues['negative_values'] == 1

    def test_detect_duplicates(self):
        """Test detecting duplicate timestamps."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-01'])
        df = pd.DataFrame({'close': [100, 101]}, index=dates)

        issues = detect_data_issues(df)

        assert issues['duplicate_timestamps'] == 1

    def test_detect_not_sorted(self):
        """Test detecting unsorted index."""
        dates = pd.to_datetime(['2024-01-02', '2024-01-01'])
        df = pd.DataFrame({'close': [100, 101]}, index=dates)

        issues = detect_data_issues(df)

        assert issues['not_sorted'] is True

    def test_detect_high_low_inconsistent(self):
        """Test detecting high < low."""
        df = pd.DataFrame({
            'high': [100, 102],
            'low': [99, 103]  # Second row: high < low
        })

        issues = detect_data_issues(df)

        assert issues['high_low_inconsistent'] == 1

    def test_detect_ohlc_invalid(self):
        """Test detecting invalid OHLC relationships."""
        df = pd.DataFrame({
            'open': [100, 100],
            'high': [102, 98],  # Second row: high < open
            'low': [99, 97],
            'close': [101, 99]
        })

        issues = detect_data_issues(df)

        assert issues['ohlc_relationship_invalid'] >= 1


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_required_columns(self):
        """Test getting required columns list."""
        required = get_required_columns()

        assert 'open' in required
        assert 'high' in required
        assert 'low' in required
        assert 'close' in required
        assert 'volume' in required

    def test_get_column_aliases(self):
        """Test getting column aliases."""
        aliases = get_column_aliases('open')

        assert 'Open' in aliases
        assert 'OPEN' in aliases

    def test_get_column_aliases_invalid(self):
        """Test getting aliases for unknown column."""
        aliases = get_column_aliases('unknown')

        assert aliases == []
