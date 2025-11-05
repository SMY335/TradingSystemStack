"""
Cryptocurrency Data Fetcher using CCXT
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import ccxt


class CryptoDataFetcher:
    """Fetch crypto OHLCV data from exchanges via CCXT"""

    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize the data fetcher

        Args:
            exchange_id: Exchange to use (e.g., 'binance', 'kraken', 'coinbase')
        """
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        days_back: int = 365,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a trading pair

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Candlestick timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            days_back: Number of days of historical data to fetch
            limit: Optional limit on number of candles (overrides days_back)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            # Calculate since timestamp
            if limit is None:
                since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            else:
                since = None

            # Fetch OHLCV data
            print(f"Fetching {symbol} data from {self.exchange_id}...")
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit or 1000
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            print(f"✓ Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")

            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def get_available_symbols(self, quote_currency: str = 'USDT') -> list[str]:
        """
        Get list of available trading pairs

        Args:
            quote_currency: Filter by quote currency (e.g., 'USDT', 'BTC')

        Returns:
            List of symbol strings
        """
        try:
            markets = self.exchange.load_markets()
            symbols = [
                symbol for symbol in markets.keys()
                if quote_currency in symbol and markets[symbol]['active']
            ]
            return sorted(symbols)
        except Exception as e:
            print(f"Error loading markets: {e}")
            return []

    @staticmethod
    def get_supported_exchanges() -> list[str]:
        """Get list of supported exchanges"""
        return ['binance', 'kraken', 'coinbase', 'bybit', 'okx', 'kucoin']


# Example usage
if __name__ == "__main__":
    fetcher = CryptoDataFetcher('binance')
    df = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', days_back=30)
    print(df.head())
    print(df.tail())
