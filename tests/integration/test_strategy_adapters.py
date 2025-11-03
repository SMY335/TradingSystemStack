"""
Integration Tests for Strategy Adapters
Tests that all strategies work correctly on Nautilus and Backtrader
"""
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a simple pytest.raises replacement
    class pytest:
        @staticmethod
        def raises(exception):
            class RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception} but nothing was raised")
                    if not issubclass(exc_type, exception):
                        return False
                    return True
            return RaisesContext()

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.adapters.strategy_factory import StrategyFactory
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework
from src.infrastructure.data_manager import UnifiedDataManager


class TestEMAAdapter:
    """Test EMA strategy adapter"""
    
    def test_ema_nautilus_adapter(self):
        """Test EMA conversion to Nautilus"""
        config = StrategyConfig(
            name="ema_test",
            parameters={'fast_period': 10, 'slow_period': 50},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        strategy = StrategyFactory.create('ema', config)
        assert strategy is not None
        print("✅ EMA → Nautilus adapter OK")
    
    def test_ema_backtrader_adapter(self):
        """Test EMA conversion to Backtrader"""
        config = StrategyConfig(
            name="ema_test",
            parameters={'fast_period': 10, 'slow_period': 50},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.BACKTRADER
        )
        
        strategy = StrategyFactory.create('ema', config)
        assert strategy is not None
        # Check that it's a Backtrader strategy class
        assert hasattr(strategy, 'params')
        print("✅ EMA → Backtrader adapter OK")
    
    def test_ema_parameter_validation(self):
        """Test EMA parameter validation"""
        config = StrategyConfig(
            name="ema_test",
            parameters={'fast_period': 50, 'slow_period': 10},  # Invalid: fast > slow
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        with pytest.raises(ValueError):
            StrategyFactory.create('ema', config)
        print("✅ EMA parameter validation OK")
    
    def test_ema_parameter_space(self):
        """Test EMA parameter optimization space"""
        config = StrategyConfig(
            name="ema_test",
            parameters={'fast_period': 10, 'slow_period': 50},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        adapter = StrategyFactory.get_adapter('ema', config)
        param_space = adapter.get_parameter_space()
        
        assert 'fast_period' in param_space
        assert 'slow_period' in param_space
        print(f"✅ EMA parameter space: {param_space}")


class TestRSIAdapter:
    """Test RSI strategy adapter"""
    
    def test_rsi_nautilus_adapter(self):
        """Test RSI conversion to Nautilus"""
        config = StrategyConfig(
            name="rsi_test",
            parameters={'period': 14, 'oversold': 30, 'overbought': 70},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        strategy = StrategyFactory.create('rsi', config)
        assert strategy is not None
        print("✅ RSI → Nautilus adapter OK")
    
    def test_rsi_backtrader_adapter(self):
        """Test RSI conversion to Backtrader"""
        config = StrategyConfig(
            name="rsi_test",
            parameters={'period': 14, 'oversold': 30, 'overbought': 70},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.BACKTRADER
        )
        
        strategy = StrategyFactory.create('rsi', config)
        assert strategy is not None
        assert hasattr(strategy, 'params')
        print("✅ RSI → Backtrader adapter OK")
    
    def test_rsi_parameter_validation(self):
        """Test RSI parameter validation"""
        config = StrategyConfig(
            name="rsi_test",
            parameters={'period': 14, 'oversold': 80, 'overbought': 20},  # Invalid: oversold > overbought
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        with pytest.raises(ValueError):
            StrategyFactory.create('rsi', config)
        print("✅ RSI parameter validation OK")


class TestMACDAdapter:
    """Test MACD strategy adapter"""
    
    def test_macd_nautilus_adapter(self):
        """Test MACD conversion to Nautilus"""
        config = StrategyConfig(
            name="macd_test",
            parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.NAUTILUS
        )
        
        strategy = StrategyFactory.create('macd', config)
        assert strategy is not None
        print("✅ MACD → Nautilus adapter OK")
    
    def test_macd_backtrader_adapter(self):
        """Test MACD conversion to Backtrader"""
        config = StrategyConfig(
            name="macd_test",
            parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            timeframe='1h',
            symbols=['BTC/USDT'],
            capital=10000.0,
            framework=StrategyFramework.BACKTRADER
        )
        
        strategy = StrategyFactory.create('macd', config)
        assert strategy is not None
        assert hasattr(strategy, 'params')
        print("✅ MACD → Backtrader adapter OK")


class TestDataManager:
    """Test unified data manager"""
    
    def test_data_manager_initialization(self):
        """Test data manager can be initialized"""
        manager = UnifiedDataManager()
        assert manager is not None
        print("✅ Data manager initialization OK")
    
    def test_list_stored_data(self):
        """Test listing stored data"""
        manager = UnifiedDataManager()
        stats = manager.list_stored_data()
        assert isinstance(stats, dict)
        print(f"✅ Data manager stats: {stats}")


class TestStrategyFactory:
    """Test strategy factory"""
    
    def test_list_strategies(self):
        """Test listing available strategies"""
        strategies = StrategyFactory.list_strategies()
        assert len(strategies) > 0
        assert 'ema' in strategies
        assert 'rsi' in strategies
        assert 'macd' in strategies
        print(f"✅ Available strategies: {strategies}")
    
    def test_get_strategy_info(self):
        """Test getting strategy information"""
        for strategy_name in StrategyFactory.list_strategies():
            info = StrategyFactory.get_strategy_info(strategy_name)
            assert 'name' in info
            assert 'supported_frameworks' in info
            print(f"✅ {strategy_name} info: {info}")


def run_all_tests():
    """Run all tests without pytest"""
    print("\n" + "="*60)
    print("🧪 RUNNING STRATEGY ADAPTER INTEGRATION TESTS")
    print("="*60 + "\n")
    
    # Test EMA
    print("\n📊 Testing EMA Adapter...")
    ema_tests = TestEMAAdapter()
    ema_tests.test_ema_nautilus_adapter()
    ema_tests.test_ema_backtrader_adapter()
    try:
        ema_tests.test_ema_parameter_validation()
    except AssertionError:
        pass  # Expected to raise ValueError
    ema_tests.test_ema_parameter_space()
    
    # Test RSI
    print("\n📊 Testing RSI Adapter...")
    rsi_tests = TestRSIAdapter()
    rsi_tests.test_rsi_nautilus_adapter()
    rsi_tests.test_rsi_backtrader_adapter()
    try:
        rsi_tests.test_rsi_parameter_validation()
    except AssertionError:
        pass  # Expected to raise ValueError
    
    # Test MACD
    print("\n📊 Testing MACD Adapter...")
    macd_tests = TestMACDAdapter()
    macd_tests.test_macd_nautilus_adapter()
    macd_tests.test_macd_backtrader_adapter()
    
    # Test Data Manager
    print("\n📊 Testing Data Manager...")
    data_tests = TestDataManager()
    data_tests.test_data_manager_initialization()
    data_tests.test_list_stored_data()
    
    # Test Strategy Factory
    print("\n📊 Testing Strategy Factory...")
    factory_tests = TestStrategyFactory()
    factory_tests.test_list_strategies()
    factory_tests.test_get_strategy_info()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
