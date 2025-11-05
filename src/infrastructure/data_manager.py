"""
Unified Data Manager for TradingSystemStack
Manages data fetching from CCXT and storage in ArcticDB
Provides conversion to Nautilus and Backtrader formats
"""
import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from src.infrastructure.arctic_manager import ArcticManager

# Configure logging
logger = logging.getLogger(__name__)

# Valid timeframes across exchanges
VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

# Nautilus imports
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.objects import Price, Quantity

# Backtrader imports
import backtrader as bt


class BacktraderPandasData(bt.feeds.PandasData):
    """Custom Backtrader data feed from pandas DataFrame"""
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', -1),
    )


class UnifiedDataManager:
    """
    Centralized data management for all trading frameworks
    Handles CCXT fetching, ArcticDB storage, and format conversion
    """
    
    def __init__(self, arctic_path: str = "data/arctic_db"):
        """
        Initialize data manager
        
        Args:
            arctic_path: Path to ArcticDB storage
        """
        # Validate arctic_path
        if not arctic_path or not isinstance(arctic_path, str):
            raise ValueError("arctic_path must be a non-empty string")
        
        self.arctic_manager = ArcticManager(arctic_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        logger.info(f"UnifiedDataManager initialized with arctic_path: {arctic_path}")
        
    def _get_exchange(self, exchange_name: str) -> ccxt.Exchange:
        """
        Get or create CCXT exchange instance
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            
        Returns:
            CCXT exchange instance
        """
        # Validate exchange_name
        if not exchange_name or not isinstance(exchange_name, str):
            raise ValueError("exchange_name must be a non-empty string")
        
        # Check if exchange is supported by CCXT
        if not hasattr(ccxt, exchange_name):
            available_exchanges = ccxt.exchanges[:10]  # Show first 10 for clarity
            raise ValueError(
                f"Exchange '{exchange_name}' not supported by CCXT. "
                f"Available exchanges include: {available_exchanges}..."
            )
        
        if exchange_name not in self.exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchanges[exchange_name] = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info(f"Created exchange instance: {exchange_name}")
            except Exception as e:
                logger.error(f"Failed to create exchange {exchange_name}: {e}")
                raise ConnectionError(f"Failed to initialize exchange '{exchange_name}'") from e
        
        return self.exchanges[exchange_name]
    
    def fetch_and_store(
        self, 
        exchange: str, 
        symbol: str,
        timeframe: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange and store in ArcticDB
        
        Args:
            exchange: Exchange name (e.g., 'binance')
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date for data
            end_date: End date for data (default: now)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        
        if '/' not in symbol:
            raise ValueError(f"Invalid symbol format '{symbol}'. Expected format: 'BTC/USDT'")
        
        # Validate timeframe
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        # Validate start_date
        if not isinstance(start_date, datetime):
            raise TypeError(f"start_date must be datetime, got {type(start_date)}")
        
        # Set default end_date if not provided
        if end_date is None:
            end_date = datetime.now()
        
        # Validate end_date
        if not isinstance(end_date, datetime):
            raise TypeError(f"end_date must be datetime, got {type(end_date)}")
        
        # Validate date range
        if start_date >= end_date:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )
        
        # Check if end_date is in the future
        if end_date > datetime.now():
            raise ValueError(f"end_date ({end_date}) cannot be in the future")
        
        # Check reasonable date range (max 2 years)
        max_days = 730  # ~2 years
        date_diff = (end_date - start_date).days
        if date_diff > max_days:
            raise ValueError(
                f"Date range too large: {date_diff} days. Maximum allowed: {max_days} days"
            )
        
        logger.info(f"Fetching {symbol} {timeframe} from {exchange} ({start_date} to {end_date})")
        
        print(f"📥 Fetching {symbol} {timeframe} from {exchange}...")
        
        # Get exchange instance
        exch = self._get_exchange(exchange)
        
        # Convert dates to timestamps
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        # Fetch data in chunks
        all_data = []
        current_since = since
        
        while current_since < until:
            try:
                ohlcv = exch.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_since,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update timestamp for next batch
                current_since = ohlcv[-1][0] + 1
                
                # Stop if we've reached the end date
                if current_since >= until:
                    break
                    
            except Exception as e:
                print(f"❌ Error fetching data: {e}")
                break
        
        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            print(f"⚠️  No data fetched for {symbol}")
            raise ValueError(f"No data returned for {symbol} {timeframe} from {exchange}")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Store in Arctic with metadata
        storage_key = f"{symbol.replace('/', '_')}_{timeframe}"
        metadata = {
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'fetched_at': datetime.now().isoformat()
        }
        
        self.arctic_manager.write_market_data(storage_key, df, metadata=metadata)
        
        print(f"✅ Stored {len(df)} bars for {symbol} {timeframe}")
        return df
    
    def get_data(
        self, 
        symbol: str,
        timeframe: str, 
        start: datetime, 
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve data from ArcticDB
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start: Start date
            end: End date (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        
        if '/' not in symbol:
            raise ValueError(f"Invalid symbol format '{symbol}'. Expected format: 'BTC/USDT'")
        
        # Validate timeframe
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        # Validate start date
        if not isinstance(start, datetime):
            raise TypeError(f"start must be datetime, got {type(start)}")
        
        # Validate end date if provided
        if end is not None:
            if not isinstance(end, datetime):
                raise TypeError(f"end must be datetime, got {type(end)}")
            
            if start >= end:
                raise ValueError(f"start ({start}) must be before end ({end})")
        
        storage_key = f"{symbol.replace('/', '_')}_{timeframe}"
        
        try:
            df = self.arctic_manager.read_market_data(
                storage_key,
                start_date=start,
                end_date=end
            )
            print(f"📊 Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df
        except Exception as e:
            print(f"❌ Error retrieving data: {e}")
            return pd.DataFrame()
    
    def to_nautilus_bars(self, df: pd.DataFrame, symbol: str, timeframe: str) -> list:
        """
        Convert DataFrame to Nautilus Bar objects
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            List of Nautilus Bar objects
        """
        # Validate DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        
        # Validate timeframe
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        logger.debug(f"Converting {len(df)} bars to Nautilus format for {symbol}")
        
        bars = []
        
        # Create bar type
        symbol_clean = symbol.replace('/', '')
        bar_type = BarType.from_str(f"{symbol_clean}.BINANCE-{timeframe}-LAST")
        
        for timestamp, row in df.iterrows():
            # Convert to Nautilus Bar
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(str(row['open'])),
                high=Price.from_str(str(row['high'])),
                low=Price.from_str(str(row['low'])),
                close=Price.from_str(str(row['close'])),
                volume=Quantity.from_str(str(row['volume'])),
                ts_event=dt_to_unix_nanos(pd.Timestamp(timestamp)),
                ts_init=dt_to_unix_nanos(pd.Timestamp(timestamp))
            )
            bars.append(bar)
        
        print(f"✅ Converted {len(bars)} bars to Nautilus format")
        return bars
    
    def to_backtrader_feed(self, df: pd.DataFrame) -> BacktraderPandasData:
        """
        Convert DataFrame to Backtrader data feed
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Backtrader PandasData feed
        """
        # Validate DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Check minimum length for meaningful backtesting
        if len(df) < 50:
            logger.warning(f"DataFrame has only {len(df)} rows. Recommend at least 50 for meaningful results.")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create Backtrader data feed
        data_feed = BacktraderPandasData(dataname=df)
        
        print(f"✅ Created Backtrader feed with {len(df)} bars")
        return data_feed
    
    def list_stored_data(self) -> Dict[str, list]:
        """
        List all stored data in ArcticDB
        
        Returns:
            Dictionary with library statistics
        """
        return self.arctic_manager.get_library_stats()
    
    def delete_data(self, symbol: str, timeframe: str):
        """
        Delete stored data for a symbol/timeframe
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        
        # Validate timeframe
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        storage_key = f"{symbol.replace('/', '_')}_{timeframe}"
        self.arctic_manager.delete_symbol('market_data', storage_key)
        print(f"✅ Deleted data for {symbol} {timeframe}")


if __name__ == "__main__":
    # Example usage
    manager = UnifiedDataManager()
    
    # Fetch and store data
    start = datetime.now() - timedelta(days=7)
    df = manager.fetch_and_store(
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=start
    )
    
    # Retrieve data
    df = manager.get_data('BTC/USDT', '1h', start)
    
    # Convert to different formats
    if not df.empty:
        nautilus_bars = manager.to_nautilus_bars(df, 'BTC/USDT', '1h')
        bt_feed = manager.to_backtrader_feed(df)
        print("\n✅ Data manager test completed successfully!")
