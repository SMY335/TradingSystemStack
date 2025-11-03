"""
Data Migration Script
Migrates historical market data from CCXT to ArcticDB
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.data_manager import UnifiedDataManager
from datetime import datetime, timedelta
import argparse


def migrate_historical_data(days: int = 365, symbols: list = None, timeframes: list = None):
    """
    Migrate historical data from CCXT to ArcticDB
    
    Args:
        days: Number of days of historical data to fetch
        symbols: List of trading pairs (default: BTC/USDT, ETH/USDT, BNB/USDT)
        timeframes: List of timeframes (default: 1h, 4h, 1d)
    """
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    if timeframes is None:
        timeframes = ['1h', '4h', '1d']
    
    manager = UnifiedDataManager()
    start = datetime.now() - timedelta(days=days)
    
    print(f"\n{'='*60}")
    print(f"📦 STARTING DATA MIGRATION TO ARCTICDB")
    print(f"{'='*60}")
    print(f"Period: {start.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_count = 0
    
    for symbol in symbols:
        for tf in timeframes:
            try:
                print(f"\n📥 Migrating {symbol} {tf}...")
                df = manager.fetch_and_store(
                    exchange='binance',
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start
                )
                
                if not df.empty:
                    print(f"✅ {symbol} {tf} migrated successfully ({len(df)} bars)")
                    success_count += 1
                else:
                    print(f"⚠️  {symbol} {tf} returned empty data")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error migrating {symbol} {tf}: {e}")
                failed_count += 1
    
    print(f"\n{'='*60}")
    print(f"📊 MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful migrations: {success_count}")
    print(f"❌ Failed migrations: {failed_count}")
    print(f"Total: {success_count + failed_count}")
    print(f"{'='*60}\n")
    
    # Show stored data
    print("📦 Stored data in ArcticDB:")
    stats = manager.list_stored_data()
    for lib_name, lib_stats in stats.items():
        if lib_stats['symbol_count'] > 0:
            print(f"\n{lib_name}:")
            print(f"  Total symbols: {lib_stats['symbol_count']}")
            print(f"  Samples: {', '.join(lib_stats['symbols'][:5])}")


def verify_migration():
    """Verify that data was successfully migrated"""
    print("\n🔍 Verifying migration...")
    
    manager = UnifiedDataManager()
    test_symbols = [
        ('BTC/USDT', '1h'),
        ('ETH/USDT', '4h'),
        ('BNB/USDT', '1d')
    ]
    
    all_ok = True
    for symbol, timeframe in test_symbols:
        try:
            start = datetime.now() - timedelta(days=7)
            df = manager.get_data(symbol, timeframe, start)
            
            if not df.empty:
                print(f"✅ {symbol} {timeframe}: {len(df)} bars retrieved")
            else:
                print(f"⚠️  {symbol} {timeframe}: No data found")
                all_ok = False
        except Exception as e:
            print(f"❌ {symbol} {timeframe}: Error - {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ All verifications passed!")
    else:
        print("\n⚠️  Some verifications failed")
    
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Migrate market data to ArcticDB')
    parser.add_argument(
        '--days', 
        type=int, 
        default=365, 
        help='Number of days of historical data to fetch (default: 365)'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        help='Trading pairs to migrate (default: BTC/USDT ETH/USDT BNB/USDT)'
    )
    parser.add_argument(
        '--timeframes', 
        nargs='+', 
        help='Timeframes to migrate (default: 1h 4h 1d)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data without fetching new data'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_migration()
    else:
        migrate_historical_data(
            days=args.days,
            symbols=args.symbols,
            timeframes=args.timeframes
        )
        
        # Verify after migration
        print("\n" + "="*60)
        verify_migration()
