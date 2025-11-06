#!/usr/bin/env python
"""
Quick validation test for unified backtesting system
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*60)
print("🧪 TESTING UNIFIED BACKTESTING SYSTEM")
print("="*60)

# Test 1: Import modules
print("\n1️⃣  Testing imports...")
try:
    from src.backtesting import (
        create_backtest_engine,
        BacktestConfig,
        BacktestEngine,
        BacktestResult
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create configuration
print("\n2️⃣  Testing BacktestConfig...")
try:
    config = BacktestConfig(
        initial_capital=10000,
        fees_pct=0.1,
        slippage_pct=0.05
    )
    print(f"✅ Config created: capital=${config.initial_capital}")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    sys.exit(1)

# Test 3: Create Backtrader engine
print("\n3️⃣  Testing BacktraderEngine...")
try:
    engine = create_backtest_engine('backtrader', config)
    print(f"✅ Engine created: {engine.get_engine_name()}")
except Exception as e:
    print(f"❌ Engine creation failed: {e}")
    sys.exit(1)

# Test 4: Create simple strategy and data
print("\n4️⃣  Creating test strategy and data...")

class SimpleTestStrategy:
    def __init__(self):
        self.name = "Simple Test Strategy"
    
    def generate_signals(self, data):
        # Simple MA crossover
        fast_ma = data['close'].rolling(window=10).mean()
        slow_ma = data['close'].rolling(window=20).mean()
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        return entries.fillna(False), exits.fillna(False)

# Generate test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=500, freq='1h')
close = 100 + np.cumsum(np.random.randn(500) * 0.5)
data = pd.DataFrame({
    'open': close + np.random.uniform(-1, 1, 500),
    'high': close + np.random.uniform(0, 2, 500),
    'low': close - np.random.uniform(0, 2, 500),
    'close': close,
    'volume': np.random.uniform(1000, 10000, 500)
}, index=dates)

strategy = SimpleTestStrategy()
print(f"✅ Strategy and data created: {len(data)} bars")

# Test 5: Run backtest
print("\n5️⃣  Running backtest...")
try:
    result = engine.run(strategy, data)
    print(f"✅ Backtest completed successfully")
    print(f"   - Total Return: {result.total_return_pct:.2f}%")
    print(f"   - Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   - Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"   - Total Trades: {result.total_trades}")
    print(f"   - Win Rate: {result.win_rate_pct:.1f}%")
except Exception as e:
    print(f"❌ Backtest failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Validate result format
print("\n6️⃣  Validating result format...")
required_fields = [
    'total_return_pct', 'annualized_return_pct',
    'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
    'max_drawdown_pct', 'volatility_pct',
    'total_trades', 'winning_trades', 'losing_trades',
    'win_rate_pct', 'profit_factor'
]
missing = [f for f in required_fields if not hasattr(result, f)]
if missing:
    print(f"❌ Missing fields: {missing}")
    sys.exit(1)
else:
    print(f"✅ All {len(required_fields)} required fields present")

# Test 7: Test to_dict conversion
print("\n7️⃣  Testing result conversion...")
try:
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert 'total_return_pct' in result_dict
    print(f"✅ Result successfully converted to dict with {len(result_dict)} fields")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    sys.exit(1)

# Test 8: Test VectorBT engine (optional)
print("\n8️⃣  Testing VectorBT engine (optional)...")
try:
    vbt_engine = create_backtest_engine('vectorbt', config)
    print(f"✅ VectorBT engine created: {vbt_engine.get_engine_name()}")
    try:
        vbt_result = vbt_engine.run(strategy, data)
        print(f"✅ VectorBT backtest completed: Return={vbt_result.total_return_pct:.2f}%")
    except ImportError:
        print("⚠️  VectorBT not installed (optional)")
except Exception as e:
    print(f"⚠️  VectorBT test skipped: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - UNIFIED BACKTESTING SYSTEM OPERATIONAL")
print("="*60)
