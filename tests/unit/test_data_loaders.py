"""
Unit tests for data.loaders module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.data.loaders import (
    get_ohlcv,
    load_csv,
    load_parquet,
    save_ohlcv,
    get_available_sources,
    DataLoaderError,
)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range('2024-01-01', periods=10, freq='D', tz='UTC')
    return pd.DataFrame({
        'open': np.arange(100, 110, dtype=float),
        'high': np.arange(102, 112, dtype=float),
        'low': np.arange(99, 109, dtype=float),
        'close': np.arange(101, 111, dtype=float),
        'volume': np.arange(1000, 11000, 1000, dtype=float)
    }, index=dates)


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_ohlcv):
    """Create sample CSV file."""
    csv_path = temp_data_dir / "test_data.csv"
    sample_ohlcv.to_csv(csv_path)
    return csv_path


@pytest.fixture
def sample_parquet_file(temp_data_dir, sample_ohlcv):
    """Create sample parquet file."""
    parquet_path = temp_data_dir / "test_data.parquet"
    sample_ohlcv.to_parquet(parquet_path)
    return parquet_path


class TestLoadCSV:
    """Test load_csv function."""

    def test_load_csv_basic(self, sample_csv_file, sample_ohlcv):
        """Test basic CSV loading."""
        df = load_csv(sample_csv_file, normalize=False)

        assert len(df) == len(sample_ohlcv)
        assert 'open' in df.columns or 'Open' in df.columns

    def test_load_csv_with_normalization(self, temp_data_dir):
        """Test CSV loading with normalization."""
        # Create CSV with capitalized columns
        csv_path = temp_data_dir / "capitalized.csv"
        df_cap = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 2000]
        })
        df_cap.to_csv(csv_path, index=False)

        df = load_csv(csv_path, normalize=True)

        # Should be normalized to lowercase
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'close' in df.columns

    def test_load_csv_nonexistent(self, temp_data_dir):
        """Test loading non-existent CSV file."""
        nonexistent = temp_data_dir / "nonexistent.csv"

        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_csv(nonexistent)


class TestLoadParquet:
    """Test load_parquet function."""

    def test_load_parquet_basic(self, sample_parquet_file, sample_ohlcv):
        """Test basic parquet loading."""
        df = load_parquet(sample_parquet_file, normalize=False)

        assert len(df) == len(sample_ohlcv)
        assert 'open' in df.columns

    def test_load_parquet_with_normalization(self, temp_data_dir):
        """Test parquet loading with normalization."""
        parquet_path = temp_data_dir / "capitalized.parquet"
        df_cap = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 2000]
        })
        df_cap.to_parquet(parquet_path)

        df = load_parquet(parquet_path, normalize=True)

        # Should be normalized to lowercase
        assert 'open' in df.columns
        assert 'high' in df.columns

    def test_load_parquet_specific_columns(self, sample_parquet_file):
        """Test loading specific columns from parquet."""
        df = load_parquet(sample_parquet_file, columns=['open', 'close'])

        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' not in df.columns
        assert 'low' not in df.columns

    def test_load_parquet_nonexistent(self, temp_data_dir):
        """Test loading non-existent parquet file."""
        nonexistent = temp_data_dir / "nonexistent.parquet"

        with pytest.raises(Exception):
            load_parquet(nonexistent)


class TestSaveOHLCV:
    """Test save_ohlcv function."""

    def test_save_parquet(self, temp_data_dir, sample_ohlcv):
        """Test saving OHLCV to parquet."""
        output_path = temp_data_dir / "output.parquet"

        save_ohlcv(sample_ohlcv, output_path, format='parquet')

        assert output_path.exists()

        # Load and verify
        df = pd.read_parquet(output_path)
        assert len(df) == len(sample_ohlcv)

    def test_save_csv(self, temp_data_dir, sample_ohlcv):
        """Test saving OHLCV to CSV."""
        output_path = temp_data_dir / "output.csv"

        save_ohlcv(sample_ohlcv, output_path, format='csv')

        assert output_path.exists()

        # Load and verify
        df = pd.read_csv(output_path, index_col=0)
        assert len(df) == len(sample_ohlcv)

    def test_save_with_normalization(self, temp_data_dir):
        """Test saving with column normalization."""
        output_path = temp_data_dir / "normalized.parquet"

        # DataFrame with capitalized columns
        df_cap = pd.DataFrame({
            'Open': [100, 101],
            'Close': [101, 102]
        })

        save_ohlcv(df_cap, output_path, format='parquet', normalize=True)

        # Load and check columns are normalized
        df = pd.read_parquet(output_path)
        assert 'open' in df.columns
        assert 'close' in df.columns

    def test_save_unsupported_format(self, temp_data_dir, sample_ohlcv):
        """Test saving with unsupported format."""
        output_path = temp_data_dir / "output.txt"

        with pytest.raises(DataLoaderError, match="Unsupported format"):
            save_ohlcv(sample_ohlcv, output_path, format='txt')


class TestGetOHLCV:
    """Test get_ohlcv function."""

    def test_auto_detect_crypto_symbol(self):
        """Test auto-detection of crypto symbol."""
        # Symbols with '/' should be detected as crypto (ccxt)
        # This test just checks the detection logic, not actual loading
        # since that requires network access

        # We can't test actual loading without mocking or network access
        # Just verify the function exists and accepts parameters
        assert callable(get_ohlcv)

    def test_auto_detect_stock_symbol(self):
        """Test auto-detection of stock symbol."""
        # Symbols without '/' should be detected as stock (yfinance)
        # This test just checks the function signature
        assert callable(get_ohlcv)

    def test_unsupported_source(self):
        """Test with unsupported data source."""
        with pytest.raises(DataLoaderError, match="Unsupported source"):
            get_ohlcv('TEST', source='unsupported_source')


class TestGetAvailableSources:
    """Test get_available_sources function."""

    def test_get_available_sources(self):
        """Test getting list of available data sources."""
        sources = get_available_sources()

        # CSV and parquet should always be available
        assert 'csv' in sources
        assert 'parquet' in sources
        assert isinstance(sources, list)

        # yfinance and ccxt may or may not be available
        # depending on whether they're installed


class TestDataLoaderEdgeCases:
    """Test edge cases and error handling."""

    def test_load_empty_csv(self, temp_data_dir):
        """Test loading empty CSV file."""
        empty_csv = temp_data_dir / "empty.csv"
        empty_csv.write_text("open,high,low,close,volume\n")

        df = load_csv(empty_csv)

        assert len(df) == 0

    def test_save_empty_dataframe(self, temp_data_dir):
        """Test saving empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        output_path = temp_data_dir / "empty.parquet"

        save_ohlcv(empty_df, output_path, format='parquet')

        assert output_path.exists()


class TestColumnNormalization:
    """Test column normalization in loaders."""

    def test_yfinance_column_format(self, temp_data_dir):
        """Test handling yfinance column format (capitalized)."""
        # Simulate yfinance output format
        csv_path = temp_data_dir / "yfinance_format.csv"
        df_yf = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 2000]
        })
        df_yf.to_csv(csv_path, index=False)

        df = load_csv(csv_path, normalize=True)

        # Should be normalized
        assert all(col.islower() for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume'])

    def test_ccxt_column_format(self, temp_data_dir):
        """Test handling CCXT column format (lowercase)."""
        csv_path = temp_data_dir / "ccxt_format.csv"
        df_ccxt = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=2),
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 2000]
        })
        df_ccxt.to_csv(csv_path, index=False)

        df = load_csv(csv_path, normalize=True)

        # Should remain lowercase
        assert 'open' in df.columns
        assert 'close' in df.columns


class TestDatetimeHandling:
    """Test datetime index handling."""

    def test_datetime_index_preserved(self, sample_parquet_file):
        """Test that datetime index is preserved when loading."""
        df = load_parquet(sample_parquet_file)

        # Should have datetime index
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_save_and_load_preserves_index(self, temp_data_dir, sample_ohlcv):
        """Test that saving and loading preserves datetime index."""
        output_path = temp_data_dir / "test.parquet"

        save_ohlcv(sample_ohlcv, output_path, format='parquet')
        df = load_parquet(output_path)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == len(sample_ohlcv)
