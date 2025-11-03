"""
ArcticDB Manager for TradingSystemStack
Manages time-series data storage for market data, order books, trades, and backtest results
"""
from arcticdb import Arctic
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime


class ArcticManager:
    """
    Manages ArcticDB libraries for different data types in the trading system
    """
    
    def __init__(self, db_path: str = "data/arctic_db"):
        """
        Initialize ArcticDB connection and libraries
        
        Args:
            db_path: Path to ArcticDB storage directory
        """
        # Ensure the directory exists
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize Arctic with LMDB storage
        self.arctic = Arctic(f"lmdb://{db_path}")
        self._init_libraries()
    
    def _init_libraries(self):
        """Initialize all required libraries if they don't exist"""
        libraries = {
            'market_data': 'OHLCV market data from exchanges',
            'orderbook': 'Order book snapshots',
            'trades': 'Trade execution records',
            'backtest_results': 'Backtesting results and metrics'
        }
        
        existing_libraries = self.arctic.list_libraries()
        
        for lib_name, description in libraries.items():
            if lib_name not in existing_libraries:
                self.arctic.create_library(lib_name)
                print(f"✅ Created library: {lib_name} - {description}")
            else:
                print(f"ℹ️  Library already exists: {lib_name}")
    
    def get_library(self, name: str):
        """
        Get a specific library by name
        
        Args:
            name: Library name (market_data, orderbook, trades, backtest_results)
            
        Returns:
            ArcticDB library object
        """
        return self.arctic[name]
    
    def write_market_data(self, symbol: str, data: pd.DataFrame, metadata: Optional[dict] = None):
        """
        Write OHLCV market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            data: DataFrame with OHLCV data
            metadata: Optional metadata dictionary
        """
        lib = self.get_library('market_data')
        lib.write(symbol, data, metadata=metadata)
        print(f"✅ Wrote {len(data)} rows for {symbol} to market_data")
    
    def read_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Read OHLCV market data for a symbol
        
        Args:
            symbol: Trading symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with market data
        """
        lib = self.get_library('market_data')
        
        if start_date and end_date:
            return lib.read(symbol, date_range=(start_date, end_date)).data
        else:
            return lib.read(symbol).data
    
    def write_orderbook(self, symbol: str, timestamp: datetime, data: pd.DataFrame):
        """
        Write order book snapshot
        
        Args:
            symbol: Trading symbol
            timestamp: Snapshot timestamp
            data: DataFrame with order book data
        """
        lib = self.get_library('orderbook')
        snapshot_key = f"{symbol}_{timestamp.isoformat()}"
        lib.write(snapshot_key, data)
    
    def write_trades(self, strategy_name: str, trades: pd.DataFrame, metadata: Optional[dict] = None):
        """
        Write trade execution records
        
        Args:
            strategy_name: Name of the trading strategy
            trades: DataFrame with trade records
            metadata: Optional metadata
        """
        lib = self.get_library('trades')
        lib.write(strategy_name, trades, metadata=metadata)
        print(f"✅ Wrote {len(trades)} trades for {strategy_name}")
    
    def write_backtest_results(self, backtest_id: str, results: pd.DataFrame, 
                               metrics: dict, metadata: Optional[dict] = None):
        """
        Write backtest results and metrics
        
        Args:
            backtest_id: Unique identifier for the backtest
            results: DataFrame with backtest results
            metrics: Dictionary of performance metrics
            metadata: Optional metadata
        """
        lib = self.get_library('backtest_results')
        
        # Combine metadata with metrics
        full_metadata = metadata or {}
        full_metadata['metrics'] = metrics
        full_metadata['timestamp'] = datetime.now().isoformat()
        
        lib.write(backtest_id, results, metadata=full_metadata)
        print(f"✅ Wrote backtest results for {backtest_id}")
    
    def read_backtest_results(self, backtest_id: str) -> tuple:
        """
        Read backtest results
        
        Args:
            backtest_id: Unique identifier for the backtest
            
        Returns:
            Tuple of (results DataFrame, metrics dict)
        """
        lib = self.get_library('backtest_results')
        data = lib.read(backtest_id)
        return data.data, data.metadata.get('metrics', {})
    
    def list_symbols(self, library: str = 'market_data') -> list:
        """
        List all symbols in a library
        
        Args:
            library: Library name
            
        Returns:
            List of symbol names
        """
        lib = self.get_library(library)
        return lib.list_symbols()
    
    def get_library_stats(self) -> dict:
        """
        Get statistics for all libraries
        
        Returns:
            Dictionary with library statistics
        """
        stats = {}
        for lib_name in self.arctic.list_libraries():
            lib = self.get_library(lib_name)
            symbols = lib.list_symbols()
            stats[lib_name] = {
                'symbol_count': len(symbols),
                'symbols': symbols[:10] if len(symbols) > 10 else symbols  # First 10 symbols
            }
        return stats
    
    def delete_symbol(self, library: str, symbol: str):
        """
        Delete a symbol from a library
        
        Args:
            library: Library name
            symbol: Symbol to delete
        """
        lib = self.get_library(library)
        lib.delete(symbol)
        print(f"✅ Deleted {symbol} from {library}")
    
    def clear_library(self, library: str):
        """
        Clear all data from a library
        
        Args:
            library: Library name
        """
        lib = self.get_library(library)
        for symbol in lib.list_symbols():
            lib.delete(symbol)
        print(f"✅ Cleared all data from {library}")


if __name__ == "__main__":
    # Example usage
    manager = ArcticManager()
    print("\n📊 Arctic Manager Statistics:")
    stats = manager.get_library_stats()
    for lib_name, lib_stats in stats.items():
        print(f"\n{lib_name}:")
        print(f"  Symbols: {lib_stats['symbol_count']}")
        if lib_stats['symbols']:
            print(f"  Sample: {lib_stats['symbols']}")
