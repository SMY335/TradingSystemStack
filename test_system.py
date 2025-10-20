"""
Test script to verify the trading system works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("🧪 Testing Trading System Components...\n")

# Test 1: Import strategies
print("1️⃣ Testing Strategies...")
try:
    from src.strategies import EMAStrategy, RSIStrategy, MACDStrategy, AVAILABLE_STRATEGIES
    print(f"   ✓ Loaded {len(AVAILABLE_STRATEGIES)} strategies")
    for name in AVAILABLE_STRATEGIES:
        print(f"     - {name}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import data fetcher
print("\n2️⃣ Testing Data Fetcher...")
try:
    from src.data_sources import CryptoDataFetcher
    fetcher = CryptoDataFetcher('binance')
    print("   ✓ CryptoDataFetcher initialized")
    print(f"   ✓ Supported exchanges: {', '.join(CryptoDataFetcher.get_supported_exchanges())}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Import backtest engine
print("\n3️⃣ Testing Backtest Engine...")
try:
    from src.backtesting import BacktestEngine
    engine = BacktestEngine(initial_cash=10000)
    print("   ✓ BacktestEngine initialized")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Fetch sample data and run backtest
print("\n4️⃣ Testing Live Data Fetch & Backtest...")
try:
    print("   → Fetching BTC/USDT data from Binance...")
    df = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', days_back=30)
    print(f"   ✓ Fetched {len(df)} candles")
    print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

    # Test strategy
    print("\n   → Running EMA strategy backtest...")
    strategy = EMAStrategy(fast_period=12, slow_period=26)
    portfolio, kpis = engine.run(strategy, df)

    print(f"   ✓ Backtest complete!")
    print(f"     - Total Return: {kpis['total_return_pct']}%")
    print(f"     - Win Rate: {kpis['win_rate_pct']}%")
    print(f"     - Total Trades: {kpis['total_trades']}")
    print(f"     - Profit Factor: {kpis['profit_factor']}")
    print(f"     - Max Drawdown: {kpis['max_drawdown_pct']}%")
    print(f"     - Final Value: ${kpis['final_value']:,.2f}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n🚀 Your trading system is ready!")
print("\nTo launch the dashboard, run:")
print("   ./run_dashboard.sh")
print("\n   or")
print("\n   streamlit run src/dashboard/app.py")
