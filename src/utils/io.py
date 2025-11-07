"""
File I/O utilities for TradingSystemStack.

Handles reading and writing of parquet and CSV files with validation.
"""
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)


class IOError(Exception):
    """Custom exception for I/O operations."""
    pass


def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[tuple]] = None
) -> pd.DataFrame:
    """Read parquet file into DataFrame.

    Args:
        file_path: Path to parquet file
        columns: Specific columns to read (None = all)
        filters: PyArrow filters to apply

    Returns:
        DataFrame with data

    Raises:
        IOError: If file cannot be read

    Examples:
        >>> df = read_parquet('data/AAPL.parquet')
        >>> df = read_parquet('data/AAPL.parquet', columns=['close', 'volume'])
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise IOError(f"File not found: {file_path}")

        if not file_path.suffix == '.parquet':
            raise IOError(f"Expected .parquet file, got: {file_path.suffix}")

        logger.debug(f"Reading parquet: {file_path}")

        df = pd.read_parquet(
            file_path,
            columns=columns,
            filters=filters,
            engine='pyarrow'
        )

        logger.debug(f"Read {len(df)} rows, {len(df.columns)} columns")

        return df

    except Exception as e:
        logger.error(f"Failed to read parquet {file_path}: {e}")
        raise IOError(f"Failed to read parquet: {e}") from e


def write_parquet(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    compression: str = 'snappy',
    index: bool = True,
    partition_cols: Optional[List[str]] = None
) -> None:
    """Write DataFrame to parquet file.

    Args:
        df: DataFrame to write
        file_path: Output file path
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'none')
        index: Whether to write index
        partition_cols: Columns to partition by (for directory structure)

    Raises:
        IOError: If file cannot be written

    Examples:
        >>> write_parquet(df, 'data/AAPL.parquet')
        >>> write_parquet(df, 'data/prices/', partition_cols=['symbol'])
    """
    try:
        file_path = Path(file_path)

        # Create parent directory if doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Writing parquet: {file_path}")

        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression=compression,
            index=index,
            partition_cols=partition_cols
        )

        logger.debug(f"Wrote {len(df)} rows to {file_path}")

    except Exception as e:
        logger.error(f"Failed to write parquet {file_path}: {e}")
        raise IOError(f"Failed to write parquet: {e}") from e


def read_csv(
    file_path: Union[str, Path],
    parse_dates: Union[bool, List[str]] = True,
    index_col: Optional[Union[int, str]] = 0,
    **kwargs
) -> pd.DataFrame:
    """Read CSV file into DataFrame.

    Args:
        file_path: Path to CSV file
        parse_dates: Parse date columns (True = auto-detect)
        index_col: Column to use as index
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame with data

    Raises:
        IOError: If file cannot be read

    Examples:
        >>> df = read_csv('data/AAPL.csv')
        >>> df = read_csv('data/AAPL.csv', index_col='date')
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            raise IOError(f"File not found: {file_path}")

        if not file_path.suffix == '.csv':
            raise IOError(f"Expected .csv file, got: {file_path.suffix}")

        logger.debug(f"Reading CSV: {file_path}")

        df = pd.read_csv(
            file_path,
            parse_dates=parse_dates,
            index_col=index_col,
            **kwargs
        )

        logger.debug(f"Read {len(df)} rows, {len(df.columns)} columns")

        return df

    except Exception as e:
        logger.error(f"Failed to read CSV {file_path}: {e}")
        raise IOError(f"Failed to read CSV: {e}") from e


def write_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    index: bool = True,
    **kwargs
) -> None:
    """Write DataFrame to CSV file.

    Args:
        df: DataFrame to write
        file_path: Output file path
        index: Whether to write index
        **kwargs: Additional arguments for DataFrame.to_csv

    Raises:
        IOError: If file cannot be written

    Examples:
        >>> write_csv(df, 'data/AAPL.csv')
        >>> write_csv(df, 'data/AAPL.csv', index=False)
    """
    try:
        file_path = Path(file_path)

        # Create parent directory if doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Writing CSV: {file_path}")

        df.to_csv(file_path, index=index, **kwargs)

        logger.debug(f"Wrote {len(df)} rows to {file_path}")

    except Exception as e:
        logger.error(f"Failed to write CSV {file_path}: {e}")
        raise IOError(f"Failed to write CSV: {e}") from e


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for directory

    Examples:
        >>> cache_dir = ensure_directory('data/cache')
        >>> cache_dir.exists()
        True
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def list_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = False
) -> List[Path]:
    """List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (default '*' = all files)
        recursive: Search recursively if True

    Returns:
        List of Path objects

    Examples:
        >>> parquet_files = list_files('data/cache', '*.parquet')
        >>> all_files = list_files('data', '*', recursive=True)
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    # Filter to files only (exclude directories)
    files = [f for f in files if f.is_file()]

    logger.debug(f"Found {len(files)} files in {directory}")

    return files


def file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB

    Examples:
        >>> size = file_size_mb('data/AAPL.parquet')
        >>> print(f"File size: {size:.2f} MB")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return 0.0

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def validate_ohlcv_columns(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """Validate DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate
        required_cols: Required column names (default: ['open', 'high', 'low', 'close', 'volume'])

    Returns:
        True if valid

    Raises:
        ValueError: If required columns are missing

    Examples:
        >>> validate_ohlcv_columns(df)  # Standard OHLCV
        True
        >>> validate_ohlcv_columns(df, ['close', 'volume'])  # Custom subset
        True
    """
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close', 'volume']

    # Case-insensitive column matching
    df_cols_lower = [col.lower() for col in df.columns]
    missing = []

    for col in required_cols:
        if col.lower() not in df_cols_lower:
            missing.append(col)

    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    return True
