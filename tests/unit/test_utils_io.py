"""
Unit tests for utils.io module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.utils.io import (
    read_parquet,
    write_parquet,
    read_csv,
    write_csv,
    ensure_directory,
    list_files,
    file_size_mb,
    validate_ohlcv_columns,
    IOError,
)


class TestParquetIO:
    """Test parquet I/O operations."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_write_and_read_parquet(self):
        """Test writing and reading parquet files."""
        # Create sample data
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        })

        file_path = self.temp_dir / 'test.parquet'

        # Write
        write_parquet(df, file_path)
        assert file_path.exists()

        # Read
        df_read = read_parquet(file_path)
        pd.testing.assert_frame_equal(df, df_read)

    def test_write_parquet_creates_directory(self):
        """Test that write_parquet creates parent directories."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        nested_path = self.temp_dir / 'nested' / 'dir' / 'test.parquet'

        write_parquet(df, nested_path)
        assert nested_path.exists()

    def test_read_nonexistent_parquet(self):
        """Test reading non-existent parquet file."""
        with pytest.raises(IOError, match="File not found"):
            read_parquet(self.temp_dir / 'nonexistent.parquet')

    def test_read_parquet_specific_columns(self):
        """Test reading specific columns from parquet."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        file_path = self.temp_dir / 'test.parquet'
        write_parquet(df, file_path)

        df_read = read_parquet(file_path, columns=['a', 'c'])
        assert list(df_read.columns) == ['a', 'c']

    def test_parquet_compression(self):
        """Test parquet with different compressions."""
        df = pd.DataFrame({'a': range(1000)})
        file_path = self.temp_dir / 'compressed.parquet'

        for compression in ['snappy', 'gzip', 'none']:
            write_parquet(df, file_path, compression=compression)
            df_read = read_parquet(file_path)
            pd.testing.assert_frame_equal(df, df_read)


class TestCSVIO:
    """Test CSV I/O operations."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_write_and_read_csv(self):
        """Test writing and reading CSV files."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })

        file_path = self.temp_dir / 'test.csv'

        # Write
        write_csv(df, file_path)
        assert file_path.exists()

        # Read
        df_read = read_csv(file_path)
        pd.testing.assert_frame_equal(df, df_read)

    def test_read_nonexistent_csv(self):
        """Test reading non-existent CSV file."""
        with pytest.raises(IOError, match="File not found"):
            read_csv(self.temp_dir / 'nonexistent.csv')

    def test_csv_with_dates(self):
        """Test CSV with date parsing."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'value': [10, 20, 30]
        })

        file_path = self.temp_dir / 'dates.csv'

        write_csv(df, file_path)
        df_read = read_csv(file_path, parse_dates=['date'])

        assert df_read['date'].dtype == 'datetime64[ns]'


class TestDirectoryOperations:
    """Test directory operations."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_ensure_directory(self):
        """Test directory creation."""
        new_dir = self.temp_dir / 'new' / 'nested' / 'dir'
        result = ensure_directory(new_dir)

        assert result.exists()
        assert result.is_dir()
        assert result == new_dir

    def test_ensure_directory_existing(self):
        """Test ensure_directory on existing directory."""
        existing = self.temp_dir / 'existing'
        existing.mkdir()

        result = ensure_directory(existing)
        assert result.exists()

    def test_list_files(self):
        """Test listing files in directory."""
        # Create test files
        (self.temp_dir / 'file1.txt').touch()
        (self.temp_dir / 'file2.txt').touch()
        (self.temp_dir / 'file3.csv').touch()

        # List all files
        all_files = list_files(self.temp_dir)
        assert len(all_files) == 3

        # List with pattern
        txt_files = list_files(self.temp_dir, '*.txt')
        assert len(txt_files) == 2

        csv_files = list_files(self.temp_dir, '*.csv')
        assert len(csv_files) == 1

    def test_list_files_recursive(self):
        """Test recursive file listing."""
        # Create nested structure
        nested_dir = self.temp_dir / 'nested'
        nested_dir.mkdir()
        (nested_dir / 'file.txt').touch()
        (self.temp_dir / 'root.txt').touch()

        # Non-recursive
        files = list_files(self.temp_dir, '*.txt', recursive=False)
        assert len(files) == 1

        # Recursive
        files = list_files(self.temp_dir, '*.txt', recursive=True)
        assert len(files) == 2

    def test_list_files_nonexistent(self):
        """Test listing files in non-existent directory."""
        files = list_files(self.temp_dir / 'nonexistent')
        assert files == []


class TestFileOperations:
    """Test file operations."""

    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up temporary directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_file_size_mb(self):
        """Test file size calculation."""
        file_path = self.temp_dir / 'test.txt'

        # Create file with known size
        content = 'x' * (1024 * 1024)  # 1MB
        file_path.write_text(content)

        size = file_size_mb(file_path)
        assert 0.9 < size < 1.1  # Allow small margin

    def test_file_size_mb_nonexistent(self):
        """Test file size of non-existent file."""
        size = file_size_mb(self.temp_dir / 'nonexistent.txt')
        assert size == 0.0


class TestOHLCVValidation:
    """Test OHLCV column validation."""

    def test_validate_valid_ohlcv(self):
        """Test validation of valid OHLCV DataFrame."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        assert validate_ohlcv_columns(df) is True

    def test_validate_case_insensitive(self):
        """Test case-insensitive column matching."""
        df = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        })

        assert validate_ohlcv_columns(df) is True

    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99]
            # Missing close and volume
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_columns(df)

    def test_validate_custom_columns(self):
        """Test validation with custom column list."""
        df = pd.DataFrame({
            'close': [100],
            'volume': [1000]
        })

        assert validate_ohlcv_columns(df, ['close', 'volume']) is True

        with pytest.raises(ValueError):
            validate_ohlcv_columns(df, ['open', 'close'])
