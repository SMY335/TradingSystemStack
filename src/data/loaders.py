"""
Unified data loading for TradingSystemStack.

Supports yfinance, CCXT, CSV, and parquet with automatic normalization.
"""
from datetime import datetime, timedelta
from typing import Optional, Union, List
from pathlib import Path
import pandas as pd
import logging

# Data sources
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from src.utils.io import read_parquet, read_csv, write_parquet
from src.utils.timeframes import normalize_timeframe
from .normalizers import normalize_columns

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loading operations."""
    pass


def get_ohlcv(
    symbol: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    interval: str = '1d',
    source: str = 'auto'
) -> pd.DataFrame:
    """Load OHLCV data from any supported source.

    Args:
        symbol: Symbol ticker (e.g., 'AAPL', 'BTC/USDT')
        start: Start date (default: 1 year ago)
        end: End date (default: now)
        interval: Timeframe (e.g., '1m', '1h', '1d')
        source: Data source ('yfinance', 'ccxt', 'auto')

    Returns:
        DataFrame with normalized OHLCV columns and DatetimeIndex

    Raises:
        DataLoaderError: If loading fails

    Examples:
        >>> # Load stock data
        >>> df = get_ohlcv('AAPL', start='2023-01-01', end='2024-01-01')

        >>> # Load crypto data
        >>> df = get_ohlcv('BTC/USDT', start='2023-01-01', source='ccxt')

        >>> # Auto-detect source
        >>> df = get_ohlcv('AAPL')  # Uses yfinance
        >>> df = get_ohlcv('BTC/USDT')  # Uses CCXT
    """
    # Normalize timeframe
    try:
        interval = normalize_timeframe(interval)
    except Exception as e:
        logger.warning(f"Could not normalize timeframe {interval}: {e}")

    # Auto-detect source
    if source == 'auto':
        if '/' in symbol:
            # Has pair separator → crypto
            source = 'ccxt'
        else:
            # No separator → stock
            source = 'yfinance'
        logger.debug(f"Auto-detected source: {source} for symbol: {symbol}")

    # Load from appropriate source
    if source == 'yfinance':
        df = load_yfinance(symbol, start, end, interval)
    elif source == 'ccxt':
        df = load_ccxt(symbol, start, end, interval)
    else:
        raise DataLoaderError(f"Unsupported source: {source}")

    # Normalize columns
    df = normalize_columns(df)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df = df.set_index('date')

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    logger.info(
        f"Loaded {len(df)} bars for {symbol} "
        f"({df.index[0]} to {df.index[-1]})"
    )

    return df


def load_yfinance(
    symbol: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    interval: str = '1d'
) -> pd.DataFrame:
    """Load data from Yahoo Finance using yfinance.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'SPY', 'TSLA')
        start: Start date
        end: End date
        interval: Timeframe ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo')

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataLoaderError: If yfinance not available or loading fails

    Examples:
        >>> df = load_yfinance('AAPL', start='2023-01-01', end='2024-01-01')
    """
    if not YFINANCE_AVAILABLE:
        raise DataLoaderError(
            "yfinance not available. Install: pip install yfinance"
        )

    # Default dates
    if start is None:
        start = datetime.now() - timedelta(days=365)
    if end is None:
        end = datetime.now()

    # Convert to datetime if string
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)

    # Map timeframe to yfinance format
    interval_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '1d': '1d',
        '1w': '1wk',
        '1M': '1mo',
    }
    yf_interval = interval_map.get(interval, interval)

    logger.debug(
        f"Loading yfinance: {symbol}, {start} to {end}, {yf_interval}"
    )

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=True  # Adjust for splits/dividends
        )

        if df.empty:
            raise DataLoaderError(f"No data returned for {symbol}")

        # yfinance returns capitalized columns: Open, High, Low, Close, Volume
        # Will be normalized by normalize_columns()

        return df

    except Exception as e:
        logger.error(f"Failed to load yfinance data: {e}")
        raise DataLoaderError(f"yfinance loading failed: {e}") from e


def load_ccxt(
    symbol: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    interval: str = '1d',
    exchange: str = 'binance'
) -> pd.DataFrame:
    """Load crypto data from exchange using CCXT.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
        start: Start date
        end: End date
        interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
        exchange: Exchange ID (default: 'binance')

    Returns:
        DataFrame with OHLCV data

    Raises:
        DataLoaderError: If CCXT not available or loading fails

    Examples:
        >>> df = load_ccxt('BTC/USDT', start='2023-01-01', exchange='binance')
    """
    if not CCXT_AVAILABLE:
        raise DataLoaderError(
            "ccxt not available. Install: pip install ccxt"
        )

    # Default dates
    if start is None:
        start = datetime.now() - timedelta(days=365)
    if end is None:
        end = datetime.now()

    # Convert to datetime if string
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)

    logger.debug(
        f"Loading CCXT: {symbol}, {start} to {end}, {interval}, {exchange}"
    )

    try:
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange)
        exchange_obj = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Convert start to timestamp (milliseconds)
        since = int(start.timestamp() * 1000)

        # Fetch OHLCV
        ohlcv = []
        current_since = since

        # Fetch in batches (CCXT has limits)
        while True:
            batch = exchange_obj.fetch_ohlcv(
                symbol,
                timeframe=interval,
                since=current_since,
                limit=1000  # Max per request
            )

            if not batch:
                break

            ohlcv.extend(batch)

            # Check if we reached end date
            last_timestamp = batch[-1][0]
            if last_timestamp >= int(end.timestamp() * 1000):
                break

            # Update since for next batch
            current_since = last_timestamp + 1

            # Avoid infinite loops
            if len(ohlcv) > 100000:
                logger.warning("Fetched 100k+ bars, stopping")
                break

        if not ohlcv:
            raise DataLoaderError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Filter to exact date range
        df = df[(df.index >= start) & (df.index <= end)]

        return df

    except Exception as e:
        logger.error(f"Failed to load CCXT data: {e}")
        raise DataLoaderError(f"CCXT loading failed: {e}") from e


def load_csv(
    file_path: Union[str, Path],
    normalize: bool = True
) -> pd.DataFrame:
    """Load OHLCV data from CSV file.

    Args:
        file_path: Path to CSV file
        normalize: Normalize column names

    Returns:
        DataFrame with OHLCV data

    Examples:
        >>> df = load_csv('data/AAPL.csv')
    """
    logger.debug(f"Loading CSV: {file_path}")

    # Try to load with first column as index
    df = read_csv(file_path, parse_dates=True, index_col=0)

    # If first column name looks like OHLCV data, it shouldn't be index
    if df.index.name and df.index.name.lower() in ['open', 'high', 'low', 'close', 'volume']:
        # Reset index and reload
        df = read_csv(file_path, parse_dates=True, index_col=None)

    if normalize:
        df = normalize_columns(df)

    return df


def load_parquet(
    file_path: Union[str, Path],
    normalize: bool = True,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load OHLCV data from parquet file.

    Args:
        file_path: Path to parquet file
        normalize: Normalize column names
        columns: Specific columns to load

    Returns:
        DataFrame with OHLCV data

    Examples:
        >>> df = load_parquet('data/cache/AAPL.parquet')
    """
    logger.debug(f"Loading parquet: {file_path}")

    df = read_parquet(file_path, columns=columns)

    if normalize:
        df = normalize_columns(df)

    return df


def save_ohlcv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    format: str = 'parquet',
    normalize: bool = True
) -> None:
    """Save OHLCV data to file.

    Args:
        df: DataFrame to save
        file_path: Output file path
        format: File format ('parquet' or 'csv')
        normalize: Normalize columns before saving

    Examples:
        >>> save_ohlcv(df, 'data/cache/AAPL.parquet')
        >>> save_ohlcv(df, 'data/export/AAPL.csv', format='csv')
    """
    if normalize:
        df = normalize_columns(df)

    file_path = Path(file_path)

    if format == 'parquet':
        write_parquet(df, file_path)
    elif format == 'csv':
        from src.utils.io import write_csv
        write_csv(df, file_path)
    else:
        raise DataLoaderError(f"Unsupported format: {format}")

    logger.info(f"Saved {len(df)} bars to {file_path}")


def get_available_sources() -> List[str]:
    """Get list of available data sources.

    Returns:
        List of source names

    Examples:
        >>> sources = get_available_sources()
        >>> print(sources)
        ['yfinance', 'ccxt', 'csv', 'parquet']
    """
    sources = ['csv', 'parquet']  # Always available

    if YFINANCE_AVAILABLE:
        sources.append('yfinance')
    if CCXT_AVAILABLE:
        sources.append('ccxt')

    return sorted(sources)
