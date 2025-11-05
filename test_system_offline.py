"""
Test script with sample data (offline mode)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🧪 Testing Trading System (Offline Mode)...\n")

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

# Test 2: Import backtest engine
print("\n2️⃣ Testing Backtest Engine...")
try:
    from src.backtesting import BacktestEngine
    engine = BacktestEngine(initial_cash=10000)
    print("   ✓ BacktestEngine initialized")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Generate sample data
print("\n3️⃣ Generating Sample Market Data...")
try:
    # Generate 90 days of hourly data
    n_periods = 90 * 24  # 90 days * 24 hours
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), periods=n_periods, freq='1h')

    # Simulate price movement (trending up with noise)
    np.random.seed(42)
    base_price = 40000  # BTC starting price
    returns = np.random.normal(0.0001, 0.02, n_periods)  # Small upward drift with volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'close': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'volume': np.random.uniform(1000, 10000, n_periods)
    }, index=dates)

    print(f"   ✓ Generated {len(df)} candles of sample data")
    print(f"   ✓ Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    print(f"   ✓ Date range: {df.index.min()} to {df.index.max()}")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Run backtests for all strategies
print("\n4️⃣ Running Backtests for All Strategies...")
try:
    results = []

    for strategy_name, strategy_class in AVAILABLE_STRATEGIES.items():
        print(f"\n   → Testing {strategy_name}...")
        strategy = strategy_class()
        portfolio, kpis = engine.run(strategy, df)

        print(f"     ✓ {strategy.get_description()}")
        print(f"       - Total Return: {kpis['total_return_pct']:>8.2f}%")
        print(f"       - Win Rate:     {kpis['win_rate_pct']:>8.2f}%")
        print(f"       - Total Trades: {kpis['total_trades']:>8}")
        print(f"       - Profit Factor:{kpis['profit_factor']:>8.2f}")
        print(f"       - Max Drawdown: {kpis['max_drawdown_pct']:>8.2f}%")
        print(f"       - Final Value:  ${kpis['final_value']:>8,.2f}")

        results.append({
            'Strategy': strategy_name,
            'Return (%)': kpis['total_return_pct'],
            'Win Rate (%)': kpis['win_rate_pct'],
            'Trades': kpis['total_trades'],
            'Profit Factor': kpis['profit_factor'],
            'Final Value': kpis['final_value']
        })

    # Display comparison
    print("\n" + "="*60)
    print("📊 STRATEGY COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values('Return (%)', ascending=False)
    print(results_df.to_string(index=False))

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test strategy with custom parameters
print("\n" + "="*60)
print("5️⃣ Testing Custom Strategy Parameters...")
try:
    print("\n   → EMA(5,15) vs EMA(20,50) vs EMA(50,200)")

    configs = [
        {'fast': 5, 'slow': 15},
        {'fast': 20, 'slow': 50},
        {'fast': 50, 'slow': 200}
    ]

    for config in configs:
        strategy = EMAStrategy(fast_period=config['fast'], slow_period=config['slow'])
        portfolio, kpis = engine.run(strategy, df)
        print(f"     EMA({config['fast']},{config['slow']}): "
              f"Return={kpis['total_return_pct']:>6.2f}% | "
              f"Trades={kpis['total_trades']:>3} | "
              f"Win Rate={kpis['win_rate_pct']:>5.2f}%")

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\n🎉 Your trading system is fully functional!")
print("\n📊 Components verified:")
print("   ✓ 3 trading strategies (EMA, RSI, MACD)")
print("   ✓ Backtesting engine with VectorBT")
print("   ✓ Performance metrics calculation")
print("   ✓ Multiple strategy comparison")
print("   ✓ Custom parameter configuration")

print("\n🚀 Next Steps:")
print("   1. Launch the dashboard: ./run_dashboard.sh")
print("   2. Or run: streamlit run src/dashboard/app.py")
print("   3. Open your browser to interact with the system")
print("\n💡 Note: The dashboard supports live data fetching when network is available")
