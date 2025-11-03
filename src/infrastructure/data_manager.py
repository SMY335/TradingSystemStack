"""
Unified Data Manager for TradingSystemStack
Manages data fetching from CCXT and storage in ArcticDB
Provides conversion to Nautilus and Backtrader formats
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from src.infrastructure.arctic_manager import ArcticManager

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
        self.arctic_manager = ArcticManager(arctic_path)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        
    def _get_exchange(self, exchange_name: str) -> ccxt.Exchange:
        """
        Get or create CCXT exchange instance
        
        Args:
            exchange_name: Name of the exchange (e.g., 'binance')
            
        Returns:
            CCXT exchange instance
        """
        if exchange_name not in self.exchanges:
            exchange_class = getattr(ccxt, exchange_name)
            self.exchanges[exchange_name] = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
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
        if end_date is None:
            end_date = datetime.now()
        
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
            print(f"⚠️  No data fetched for {symbol}")
            return pd.DataFrame()
        
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
