"""
Tests unitaires pour Advanced Backtest Engine
"""
import sys
from pathlib import Path

# Ajouter le root du projet au path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from src.backtesting.advanced_engine import BacktestConfig, TransactionCosts


def test_backtest_config_creation():
    """Test création de configuration backtest"""
    config = BacktestConfig(
        strategy_name='ema',
        strategy_params={'fast_period': 10, 'slow_period': 50},
        symbols=['BTC/USDT'],
        timeframe='1h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1),
        initial_capital=10000.0
    )
    
    assert config.strategy_name == 'ema'
    assert config.initial_capital == 10000.0
    assert config.symbols == ['BTC/USDT']
    assert config.timeframe == '1h'
    assert len(config.strategy_params) == 2
    
    print("✅ BacktestConfig créé avec succès")
    print(f"   Strategy: {config.strategy_name}")
    print(f"   Capital: {config.initial_capital}")
    print(f"   Période: {config.start_date.date()} → {config.end_date.date()}")
    
    return True


def test_transaction_costs():
    """Test modèle de coûts de transaction"""
    costs = TransactionCosts(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_basis_points=5
    )
    
    assert costs.maker_fee == 0.0002
    assert costs.taker_fee == 0.0004
    assert costs.slippage_basis_points == 5
    
    print("✅ TransactionCosts créé avec succès")
    print(f"   Maker fee: {costs.maker_fee*100}%")
    print(f"   Taker fee: {costs.taker_fee*100}%")
    print(f"   Slippage: {costs.slippage_basis_points} bps")
    
    return True


def test_transaction_costs_defaults():
    """Test valeurs par défaut des coûts de transaction"""
    costs = TransactionCosts()
    
    assert costs.maker_fee >= 0
    assert costs.taker_fee >= 0
    assert costs.slippage_basis_points >= 0
    
    print("✅ TransactionCosts avec valeurs par défaut")
    
    return True


def test_backtest_config_validation():
    """Test validation de configuration"""
    config = BacktestConfig(
        strategy_name='test_strategy',
        strategy_params={},
        symbols=['ETH/USDT'],
        timeframe='4h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=50000.0
    )
    
    # Vérifier que end_date > start_date
    assert config.end_date > config.start_date
    
    # Vérifier que initial_capital > 0
    assert config.initial_capital > 0
    
    print("✅ Validation de configuration réussie")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTS UNITAIRES - BACKTEST ENGINE")
    print("="*60 + "\n")
    
    test_backtest_config_creation()
    test_transaction_costs()
    test_transaction_costs_defaults()
    test_backtest_config_validation()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("="*60 + "\n")
