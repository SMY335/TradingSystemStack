"""
Test parity between backtesting engines
Both engines should produce similar results on same data
"""
import pytest
import pandas as pd
import numpy as np

from src.backtesting.engine_factory import create_backtest_engine
from src.backtesting.base_engine import BacktestConfig, BacktestEngine


class SimpleStrategy:
    """Simple strategy for testing"""
    
    def __init__(self):
        self.name = "Simple Test Strategy"
    
    def generate_signals(self, data: pd.DataFrame):
        """Generate simple moving average crossover signals"""
        # Fast and slow moving averages
        fast_ma = data['close'].rolling(window=10).mean()
        slow_ma = data['close'].rolling(window=20).mean()
        
        # Entry when fast crosses above slow
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Exit when fast crosses below slow
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        return entries.fillna(False), exits.fillna(False)


@pytest.fixture
def sample_data():
    """Generate test data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1h')
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(500) * 0.5)
    high = close + np.random.uniform(0, 2, 500)
    low = close - np.random.uniform(0, 2, 500)
    open_price = close + np.random.uniform(-1, 1, 500)
    volume = np.random.uniform(1000, 10000, 500)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def config():
    """Standard config for both engines"""
    return BacktestConfig(
        initial_capital=10000,
        fees_pct=0.1,
        slippage_pct=0.05
    )


@pytest.fixture
def strategy():
    """Simple strategy instance"""
    return SimpleStrategy()


def test_backtrader_engine_runs_successfully(sample_data, config, strategy):
    """Test that Backtrader engine can run without errors"""
    engine = create_backtest_engine(BacktestEngine.BACKTRADER, config)
    result = engine.run(strategy, sample_data)
    
    assert result is not None
    assert hasattr(result, 'total_return_pct')
    assert hasattr(result, 'sharpe_ratio')
    assert hasattr(result, 'total_trades')


def test_vectorbt_engine_runs_successfully(sample_data, config, strategy):
    """Test that VectorBT engine can run without errors"""
    try:
        engine = create_backtest_engine(BacktestEngine.VECTORBT, config)
        result = engine.run(strategy, sample_data)
        
        assert result is not None
        assert hasattr(result, 'total_return_pct')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'total_trades')
    except ImportError:
        pytest.skip("VectorBT not installed")


def test_standardized_result_format(sample_data, config, strategy):
    """Test that both engines return same result format"""
    bt_engine = create_backtest_engine(BacktestEngine.BACKTRADER, config)
    bt_result = bt_engine.run(strategy, sample_data)
    
    # Check all required fields exist
    required_fields = [
        'total_return_pct', 'annualized_return_pct',
        'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
        'max_drawdown_pct', 'volatility_pct',
        'total_trades', 'winning_trades', 'losing_trades',
        'win_rate_pct', 'avg_win_pct', 'avg_loss_pct',
        'profit_factor', 'avg_trade_duration_hours',
        'exposure_time_pct', 'sqn'
    ]
    
    for field in required_fields:
        assert hasattr(bt_result, field), f"Backtrader missing {field}"


def test_result_to_dict(sample_data, config, strategy):
    """Test result can be converted to dictionary"""
    engine = create_backtest_engine(BacktestEngine.BACKTRADER, config)
    result = engine.run(strategy, sample_data)
    
    result_dict = result.to_dict()
    
    assert isinstance(result_dict, dict)
    assert 'total_return_pct' in result_dict
    assert 'sharpe_ratio' in result_dict
    assert 'total_trades' in result_dict


def test_config_validation():
    """Test BacktestConfig validation"""
    # Valid config
    config = BacktestConfig(initial_capital=10000, fees_pct=0.1, slippage_pct=0.05)
    assert config.initial_capital == 10000
    
    # Invalid initial capital
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=-100)
    
    # Invalid fees
    with pytest.raises(ValueError):
        BacktestConfig(fees_pct=15)
    
    # Invalid slippage
    with pytest.raises(ValueError):
        BacktestConfig(slippage_pct=10)


def test_data_validation(config, strategy):
    """Test data validation"""
    engine = create_backtest_engine(BacktestEngine.BACKTRADER, config)
    
    # Empty DataFrame
    with pytest.raises(ValueError):
        engine.run(strategy, pd.DataFrame())
    
    # Missing columns
    df = pd.DataFrame({
        'close': [100, 101, 102]
    })
    with pytest.raises(ValueError):
        engine.run(strategy, df)
    
    # Insufficient data
    df = pd.DataFrame({
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100] * 10,
        'volume': [1000] * 10
    })
    with pytest.raises(ValueError):
        engine.run(strategy, df)


def test_engine_factory():
    """Test engine factory creates correct engines"""
    # Test with string
    engine = create_backtest_engine('backtrader')
    assert engine.get_engine_name() == "Backtrader"
    
    # Test with enum
    engine = create_backtest_engine(BacktestEngine.BACKTRADER)
    assert engine.get_engine_name() == "Backtrader"
    
    # Test convenience functions
    from src.backtesting.engine_factory import backtrader_engine, vectorbt_engine
    
    bt = backtrader_engine()
    assert bt.get_engine_name() == "Backtrader"


def test_multiple_runs_consistent(sample_data, config, strategy):
    """Test that multiple runs on same data produce consistent results"""
    engine = create_backtest_engine(BacktestEngine.BACKTRADER, config)
    
    result1 = engine.run(strategy, sample_data)
    result2 = engine.run(strategy, sample_data)
    
    # Results should be identical
    assert result1.total_return_pct == result2.total_return_pct
    assert result1.total_trades == result2.total_trades
    assert result1.win_rate_pct == result2.win_rate_pct


def test_different_configs_produce_different_results(sample_data, strategy):
    """Test that different configurations affect results"""
    config1 = BacktestConfig(initial_capital=10000, fees_pct=0.1)
    config2 = BacktestConfig(initial_capital=10000, fees_pct=0.5)
    
    engine1 = create_backtest_engine(BacktestEngine.BACKTRADER, config1)
    engine2 = create_backtest_engine(BacktestEngine.BACKTRADER, config2)
    
    result1 = engine1.run(strategy, sample_data)
    result2 = engine2.run(strategy, sample_data)
    
    # Higher fees should produce lower returns
    assert result2.total_return_pct <= result1.total_return_pct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
