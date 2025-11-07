"""
Data normalization utilities for TradingSystemStack.

Handles column name normalization, data type conversion, and validation
across different data sources (yfinance, CCXT, CSV, etc.).
"""
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Standard column mappings
STANDARD_COLUMNS = {
    'open': ['open', 'Open', 'OPEN', 'o', 'O'],
    'high': ['high', 'High', 'HIGH', 'h', 'H'],
    'low': ['low', 'Low', 'LOW', 'l', 'L'],
    'close': ['close', 'Close', 'CLOSE', 'c', 'C', 'Adj Close', 'adj_close'],
    'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V', 'vol', 'Vol'],
}

# Optional columns that may be present
OPTIONAL_COLUMNS = {
    'timestamp': ['timestamp', 'Timestamp', 'date', 'Date', 'datetime', 'Datetime'],
    'symbol': ['symbol', 'Symbol', 'ticker', 'Ticker'],
    'dividends': ['dividends', 'Dividends', 'div'],
    'splits': ['stock_splits', 'Stock Splits', 'splits', 'Splits'],
}


def normalize_columns(
    df: pd.DataFrame,
    keep_extra: bool = True,
    rename_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Normalize DataFrame column names to standard OHLCV format.

    Args:
        df: Input DataFrame
        keep_extra: Keep non-OHLCV columns (default: True)
        rename_map: Additional custom column mappings

    Returns:
        DataFrame with normalized column names (lowercase)

    Examples:
        >>> df = pd.DataFrame({'Open': [100], 'Close': [105]})
        >>> normalized = normalize_columns(df)
        >>> list(normalized.columns)
        ['open', 'close']
    """
    if df.empty:
        return df

    df = df.copy()
    normalized_cols = {}

    # Merge custom rename map if provided
    column_map = STANDARD_COLUMNS.copy()
    if rename_map:
        for target, sources in rename_map.items():
            if target in column_map:
                column_map[target].extend(sources)
            else:
                column_map[target] = sources

    # Map columns to standard names
    for col in df.columns:
        col_lower = col.lower().strip()
        found = False

        # Check standard OHLCV columns
        for standard_name, variants in column_map.items():
            if col in variants or col_lower in [v.lower() for v in variants]:
                normalized_cols[col] = standard_name
                found = True
                break

        # Check optional columns
        if not found:
            for standard_name, variants in OPTIONAL_COLUMNS.items():
                if col in variants or col_lower in [v.lower() for v in variants]:
                    normalized_cols[col] = standard_name
                    found = True
                    break

        # Keep extra columns if requested
        if not found and keep_extra:
            normalized_cols[col] = col_lower

    # Rename columns
    df = df.rename(columns=normalized_cols)

    # Log normalization
    if normalized_cols:
        logger.debug(f"Normalized columns: {normalized_cols}")

    return df


def validate_ohlcv(
    df: pd.DataFrame,
    require_volume: bool = True,
    raise_on_missing: bool = True
) -> bool:
    """Validate DataFrame has required OHLCV columns.

    Args:
        df: DataFrame to validate
        require_volume: Require volume column (default: True)
        raise_on_missing: Raise exception if columns missing (default: True)

    Returns:
        True if valid

    Raises:
        ValueError: If required columns missing and raise_on_missing=True

    Examples:
        >>> df = pd.DataFrame({'open': [100], 'high': [105], 'low': [99],
        ...                    'close': [104], 'volume': [1000]})
        >>> validate_ohlcv(df)
        True
    """
    required = ['open', 'high', 'low', 'close']
    if require_volume:
        required.append('volume')

    missing = [col for col in required if col not in df.columns]

    if missing:
        msg = f"Missing required OHLCV columns: {missing}"
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)
        return False

    return True


def ensure_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are numeric types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with numeric OHLCV columns

    Examples:
        >>> df = pd.DataFrame({'open': ['100.0'], 'close': ['105.0']})
        >>> df_numeric = ensure_numeric_types(df)
        >>> df_numeric['open'].dtype
        dtype('float64')
    """
    df = df.copy()

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']

    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def remove_invalid_rows(
    df: pd.DataFrame,
    check_ohlc: bool = True,
    check_volume: bool = False,
    remove_zero_volume: bool = False
) -> pd.DataFrame:
    """Remove rows with invalid OHLCV data.

    Args:
        df: Input DataFrame
        check_ohlc: Remove rows with NaN in OHLC (default: True)
        check_volume: Remove rows with NaN in volume (default: False)
        remove_zero_volume: Remove rows with zero volume (default: False)

    Returns:
        DataFrame with invalid rows removed

    Examples:
        >>> df = pd.DataFrame({
        ...     'open': [100, np.nan, 102],
        ...     'close': [101, 102, 103],
        ...     'volume': [1000, 2000, 0]
        ... })
        >>> clean = remove_invalid_rows(df, remove_zero_volume=True)
        >>> len(clean)
        1
    """
    df = df.copy()
    initial_len = len(df)

    # Remove NaN in OHLC
    if check_ohlc:
        ohlc_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        df = df.dropna(subset=ohlc_cols)

    # Remove NaN in volume
    if check_volume and 'volume' in df.columns:
        df = df.dropna(subset=['volume'])

    # Remove zero volume
    if remove_zero_volume and 'volume' in df.columns:
        df = df[df['volume'] > 0]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} invalid rows ({removed/initial_len*100:.1f}%)")

    return df


def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is sorted by index (timestamp).

    Args:
        df: Input DataFrame

    Returns:
        Sorted DataFrame

    Examples:
        >>> df = pd.DataFrame(
        ...     {'close': [100, 101, 102]},
        ...     index=pd.to_datetime(['2024-01-03', '2024-01-01', '2024-01-02'])
        ... )
        >>> sorted_df = ensure_sorted(df)
        >>> str(sorted_df.index[0])
        '2024-01-01 00:00:00'
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.debug("Sorted DataFrame by index")

    return df


def remove_duplicates(
    df: pd.DataFrame,
    keep: str = 'last'
) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping first or last.

    Args:
        df: Input DataFrame
        keep: Which duplicate to keep ('first' or 'last')

    Returns:
        DataFrame with duplicates removed

    Examples:
        >>> df = pd.DataFrame(
        ...     {'close': [100, 101, 102]},
        ...     index=pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        ... )
        >>> dedupe = remove_duplicates(df)
        >>> len(dedupe)
        2
    """
    df = df.copy()
    initial_len = len(df)

    # Remove duplicates based on index
    df = df[~df.index.duplicated(keep=keep)]

    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate timestamps")

    return df


def normalize_ohlcv(
    df: pd.DataFrame,
    validate: bool = True,
    ensure_numeric: bool = True,
    remove_invalid: bool = True,
    remove_zero_volume: bool = False,
    sort: bool = True,
    deduplicate: bool = True,
    keep_extra: bool = True
) -> pd.DataFrame:
    """Complete OHLCV normalization pipeline.

    Args:
        df: Input DataFrame
        validate: Validate required columns exist
        ensure_numeric: Convert to numeric types
        remove_invalid: Remove invalid rows
        remove_zero_volume: Remove zero-volume rows
        sort: Sort by timestamp
        deduplicate: Remove duplicate timestamps
        keep_extra: Keep non-OHLCV columns

    Returns:
        Fully normalized DataFrame

    Examples:
        >>> df = pd.DataFrame({
        ...     'Open': [100, 101],
        ...     'Close': [102, 103],
        ...     'High': [103, 104],
        ...     'Low': [99, 100],
        ...     'Volume': [1000, 2000]
        ... })
        >>> normalized = normalize_ohlcv(df)
        >>> list(normalized.columns)
        ['open', 'close', 'high', 'low', 'volume']
    """
    # 1. Normalize column names
    df = normalize_columns(df, keep_extra=keep_extra)

    # 2. Validate required columns
    if validate:
        validate_ohlcv(df, require_volume='volume' in df.columns, raise_on_missing=True)

    # 3. Convert to numeric types
    if ensure_numeric:
        df = ensure_numeric_types(df)

    # 4. Remove invalid rows
    if remove_invalid:
        df = remove_invalid_rows(
            df,
            check_ohlc=True,
            check_volume=False,
            remove_zero_volume=remove_zero_volume
        )

    # 5. Sort by timestamp
    if sort:
        df = ensure_sorted(df)

    # 6. Remove duplicates
    if deduplicate:
        df = remove_duplicates(df, keep='last')

    logger.debug(f"Normalized OHLCV: {len(df)} rows, {list(df.columns)}")

    return df


def detect_data_issues(df: pd.DataFrame) -> Dict[str, any]:
    """Detect potential data quality issues.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with detected issues

    Examples:
        >>> df = pd.DataFrame({
        ...     'open': [100, np.nan, 102],
        ...     'close': [101, 102, 103],
        ...     'volume': [1000, 2000, 0]
        ... })
        >>> issues = detect_data_issues(df)
        >>> issues['nan_count']
        1
    """
    issues = {}

    # Check for NaN values
    if df.isnull().any().any():
        issues['has_nan'] = True
        issues['nan_count'] = df.isnull().sum().sum()
        issues['nan_columns'] = df.columns[df.isnull().any()].tolist()
    else:
        issues['has_nan'] = False
        issues['nan_count'] = 0

    # Check for zero volume
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            issues['zero_volume_count'] = zero_volume

    # Check for negative values
    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    negative = (df[ohlcv_cols] < 0).sum().sum()
    if negative > 0:
        issues['negative_values'] = negative

    # Check for duplicates
    if df.index.duplicated().any():
        issues['duplicate_timestamps'] = df.index.duplicated().sum()

    # Check if sorted
    if not df.index.is_monotonic_increasing:
        issues['not_sorted'] = True

    # Check high/low consistency
    if all(c in df.columns for c in ['high', 'low']):
        inconsistent = (df['high'] < df['low']).sum()
        if inconsistent > 0:
            issues['high_low_inconsistent'] = inconsistent

    # Check OHLC relationship
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        ohlc_invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        if ohlc_invalid > 0:
            issues['ohlc_relationship_invalid'] = ohlc_invalid

    return issues


def get_required_columns() -> List[str]:
    """Get list of required OHLCV columns.

    Returns:
        List of required column names

    Examples:
        >>> get_required_columns()
        ['open', 'high', 'low', 'close', 'volume']
    """
    return ['open', 'high', 'low', 'close', 'volume']


def get_column_aliases(column: str) -> List[str]:
    """Get all known aliases for a column name.

    Args:
        column: Column name

    Returns:
        List of aliases

    Examples:
        >>> get_column_aliases('open')
        ['open', 'Open', 'OPEN', 'o', 'O']
    """
    column_lower = column.lower()

    # Check standard columns
    if column_lower in STANDARD_COLUMNS:
        return STANDARD_COLUMNS[column_lower]

    # Check optional columns
    if column_lower in OPTIONAL_COLUMNS:
        return OPTIONAL_COLUMNS[column_lower]

    return []
