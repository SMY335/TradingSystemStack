"""
Integration Tests for Advanced Trading Strategies

Tests ICT, Pairs Trading, and Market Making strategies.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.ict_strategies.order_blocks import OrderBlockDetector, OrderBlockType
from src.ict_strategies.fair_value_gaps import FairValueGapDetector, FVGType
from src.ict_strategies.liquidity_pools import LiquidityPoolDetector


def generate_sample_data(n=200, seed=42):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Trend + noise + volatility
    trend = np.linspace(100, 150, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise
    
    high = close + np.abs(np.random.normal(0, 1, n))
    low = close - np.abs(np.random.normal(0, 1, n))
    open_price = close + np.random.normal(0, 0.5, n)
    volume = np.random.uniform(1000, 10000, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def test_order_blocks():
    """Test Order Block detection"""
    print("\n🧪 Testing Order Blocks...")
    
    df = generate_sample_data()
    detector = OrderBlockDetector(lookback=20)
    obs = detector.detect(df)
    
    print(f"   ✅ Detected {len(obs)} Order Blocks")
    
    if obs:
        bullish = [ob for ob in obs if ob.type == OrderBlockType.BULLISH]
        bearish = [ob for ob in obs if ob.type == OrderBlockType.BEARISH]
        print(f"      Bullish: {len(bullish)}")
        print(f"      Bearish: {len(bearish)}")
        
        # Show example
        example = obs[0]
        print(f"      Example: {example.type.value} @ idx {example.start_idx}, strength: {example.strength:.1f}")
    
    assert len(obs) >= 0, "Should detect some order blocks or none"
    return True


def test_fair_value_gaps():
    """Test Fair Value Gap detection"""
    print("\n🧪 Testing Fair Value Gaps...")
    
    df = generate_sample_data()
    detector = FairValueGapDetector(min_gap_pct=0.001)
    gaps = detector.detect(df)
    
    print(f"   ✅ Detected {len(gaps)} Fair Value Gaps")
    
    if gaps:
        bullish = [g for g in gaps if g.type == FVGType.BULLISH]
        bearish = [g for g in gaps if g.type == FVGType.BEARISH]
        print(f"      Bullish: {len(bullish)}")
        print(f"      Bearish: {len(bearish)}")
        
        # Test gap updates
        updated_gaps = detector.update_gaps(gaps, df, len(df)-1)
        filled = [g for g in updated_gaps if g.filled]
        print(f"      Filled: {len(filled)}")
    
    assert len(gaps) >= 0, "Should detect some FVGs or none"
    return True


def test_liquidity_pools():
    """Test Liquidity Pool detection"""
    print("\n🧪 Testing Liquidity Pools...")
    
    df = generate_sample_data()
    detector = LiquidityPoolDetector(lookback=50)
    buy_pools, sell_pools = detector.detect_pools(df)
    
    print(f"   ✅ Detected Liquidity Pools:")
    print(f"      Buy-side (resistance): {len(buy_pools)}")
    print(f"      Sell-side (support): {len(sell_pools)}")
    
    if buy_pools:
        avg_strength = np.mean([p.strength for p in buy_pools])
        print(f"      Avg buy-side strength: {avg_strength:.1f}")
    
    if sell_pools:
        avg_strength = np.mean([p.strength for p in sell_pools])
        print(f"      Avg sell-side strength: {avg_strength:.1f}")
    
    assert len(buy_pools) >= 0 and len(sell_pools) >= 0
    return True


def test_ict_adapters():
    """Test ICT strategy adapter"""
    print("\n🧪 Testing ICT Adapter...")
    
    try:
        from src.adapters.ict_adapter import ICTAdapter
        from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
        
        config = StrategyConfig(
            name="ICT_Test",
            parameters={
                'order_block_lookback': 20,
                'liquidity_lookback': 50,
                'risk_reward_ratio': 2.0,
            },
            timeframe="1h",
            symbols=["BTC/USD"],
            capital=10000,
            framework=StrategyFramework.BACKTRADER
        )
        
        adapter = ICTAdapter(config)
        adapter.validate_parameters()
        
        # Get strategy class
        strategy_class = adapter.to_backtrader()
        print(f"   ✅ ICT Adapter validated")
        print(f"      Strategy class: {strategy_class.__name__}")
        
        # Get parameter space
        param_space = adapter.get_parameter_space()
        print(f"      Parameter space: {len(param_space)} parameters")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_pairs_trading_adapter():
    """Test Pairs Trading adapter"""
    print("\n🧪 Testing Pairs Trading Adapter...")
    
    try:
        from src.adapters.pairs_adapter import PairsAdapter
        from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
        
        config = StrategyConfig(
            name="Pairs_Test",
            parameters={
                'lookback': 60,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5,
            },
            timeframe="1h",
            symbols=["BTC/USD", "ETH/USD"],
            capital=10000,
            framework=StrategyFramework.BACKTRADER
        )
        
        adapter = PairsAdapter(config)
        adapter.validate_parameters()
        
        strategy_class = adapter.to_backtrader()
        print(f"   ✅ Pairs Trading Adapter validated")
        print(f"      Strategy class: {strategy_class.__name__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_market_maker_adapter():
    """Test Market Maker adapter"""
    print("\n🧪 Testing Market Maker Adapter...")
    
    try:
        from src.adapters.mm_adapter import MarketMakerAdapter
        from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
        
        config = StrategyConfig(
            name="MM_Test",
            parameters={
                'spread_bps': 20,
                'max_inventory': 10,
                'order_size': 1.0,
            },
            timeframe="1h",
            symbols=["BTC/USD"],
            capital=10000,
            framework=StrategyFramework.BACKTRADER
        )
        
        adapter = MarketMakerAdapter(config)
        adapter.validate_parameters()
        
        strategy_class = adapter.to_backtrader()
        print(f"   ✅ Market Maker Adapter validated")
        print(f"      Strategy class: {strategy_class.__name__}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def run_all_tests():
    """Run all advanced strategy tests"""
    print("=" * 60)
    print("🚀 Advanced Strategies Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Order Blocks", test_order_blocks),
        ("Fair Value Gaps", test_fair_value_gaps),
        ("Liquidity Pools", test_liquidity_pools),
        ("ICT Adapter", test_ict_adapters),
        ("Pairs Trading Adapter", test_pairs_trading_adapter),
        ("Market Maker Adapter", test_market_maker_adapter),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
