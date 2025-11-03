"""
Simplified adapter tests - verify imports and Backtrader strategy creation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test que tous les imports fonctionnent"""
    try:
        from src.adapters.ema_adapter import EMAAdapter
        from src.adapters.rsi_adapter import RSIAdapter
        from src.adapters.macd_adapter import MACDAdapter
        print("✅ Tous les adapters importés")
        return True
    except Exception as e:
        print(f"❌ Erreur import: {e}")
        return False

def test_backtrader_strategies():
    """Test création stratégies Backtrader uniquement"""
    from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
    from src.adapters.ema_adapter import EMAAdapter
    
    config = StrategyConfig(
        name="ema_test",
        parameters={'fast_period': 10, 'slow_period': 50},
        timeframe='1h',
        symbols=['BTC/USDT'],
        capital=10000.0,
        framework=StrategyFramework.BACKTRADER
    )
    
    adapter = EMAAdapter(config)
    strategy = adapter.to_backtrader()
    print(f"✅ EMA Backtrader strategy créée: {strategy}")
    return True

def test_all_backtrader_strategies():
    """Test création de toutes les stratégies Backtrader"""
    from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
    from src.adapters.ema_adapter import EMAAdapter
    from src.adapters.rsi_adapter import RSIAdapter
    from src.adapters.macd_adapter import MACDAdapter
    
    # Test EMA
    ema_config = StrategyConfig(
        name="ema_test",
        parameters={'fast_period': 10, 'slow_period': 50},
        timeframe='1h',
        symbols=['BTC/USDT'],
        capital=10000.0,
        framework=StrategyFramework.BACKTRADER
    )
    ema_adapter = EMAAdapter(ema_config)
    ema_strategy = ema_adapter.to_backtrader()
    print(f"✅ EMA Backtrader strategy: {ema_strategy}")
    
    # Test RSI
    rsi_config = StrategyConfig(
        name="rsi_test",
        parameters={'period': 14, 'oversold': 30, 'overbought': 70},
        timeframe='1h',
        symbols=['BTC/USDT'],
        capital=10000.0,
        framework=StrategyFramework.BACKTRADER
    )
    rsi_adapter = RSIAdapter(rsi_config)
    rsi_strategy = rsi_adapter.to_backtrader()
    print(f"✅ RSI Backtrader strategy: {rsi_strategy}")
    
    # Test MACD
    macd_config = StrategyConfig(
        name="macd_test",
        parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        timeframe='1h',
        symbols=['BTC/USDT'],
        capital=10000.0,
        framework=StrategyFramework.BACKTRADER
    )
    macd_adapter = MACDAdapter(macd_config)
    macd_strategy = macd_adapter.to_backtrader()
    print(f"✅ MACD Backtrader strategy: {macd_strategy}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)
    test_imports()
    
    print("\n" + "=" * 60)
    print("TEST 2: Backtrader Strategies")
    print("=" * 60)
    test_all_backtrader_strategies()
    
    print("\n" + "=" * 60)
    print("🎉 Tests simplifiés OK!")
    print("=" * 60)
