"""
Tests unitaires pour Walk-Forward Optimization
"""
import sys
from pathlib import Path

# Ajouter le root du projet au path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime, timedelta
from src.optimization.walk_forward import WalkForwardOptimizer, WalkForwardConfig


def test_window_generation_rolling():
    """Test génération de fenêtres rolling window"""
    config = WalkForwardConfig(
        train_period_days=180,
        test_period_days=30,
        anchored=False
    )
    
    optimizer = WalkForwardOptimizer(
        strategy_name="ema",
        parameter_space={'fast_period': [10, 20], 'slow_period': [50, 100]},
        config=config
    )
    
    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1)
    
    windows = optimizer.generate_windows(start, end)
    
    assert len(windows) > 0, "Au moins une fenêtre doit être générée"
    
    # Vérifier structure des fenêtres
    for train_start, train_end, test_start, test_end in windows:
        assert train_end == test_start, "Train et test doivent être consécutifs"
        assert (train_end - train_start).days == 180, "Train doit être 180 jours"
        assert (test_end - test_start).days == 30, "Test doit être 30 jours"
    
    print(f"✅ {len(windows)} fenêtres rolling générées correctement")


def test_window_generation_anchored():
    """Test génération de fenêtres anchored window"""
    config = WalkForwardConfig(
        train_period_days=180,
        test_period_days=30,
        anchored=True
    )
    
    optimizer = WalkForwardOptimizer(
        strategy_name="ema",
        parameter_space={'fast_period': [10]},
        config=config
    )
    
    start = datetime(2023, 1, 1)
    end = datetime(2024, 1, 1)
    
    windows = optimizer.generate_windows(start, end)
    
    # Anchored devrait générer au moins une fenêtre
    assert len(windows) >= 1, "Au moins une fenêtre anchored doit être générée"
    
    print(f"✅ {len(windows)} fenêtre(s) anchored générée(s) correctement")


def test_parameter_combinations():
    """Test génération des combinaisons de paramètres"""
    config = WalkForwardConfig()
    
    parameter_space = {
        'fast_period': [10, 20, 30],
        'slow_period': [50, 100]
    }
    
    optimizer = WalkForwardOptimizer(
        strategy_name="ema",
        parameter_space=parameter_space,
        config=config
    )
    
    combinations = optimizer._generate_param_combinations()
    
    # 3 valeurs pour fast_period × 2 valeurs pour slow_period = 6 combinaisons
    assert len(combinations) == 6, f"Expected 6 combinations, got {len(combinations)}"
    
    # Vérifier que toutes les combinaisons sont présentes
    assert {'fast_period': 10, 'slow_period': 50} in combinations
    assert {'fast_period': 30, 'slow_period': 100} in combinations
    
    print(f"✅ {len(combinations)} combinaisons de paramètres générées")


def test_empty_parameter_space():
    """Test avec espace de paramètres vide"""
    config = WalkForwardConfig()
    
    optimizer = WalkForwardOptimizer(
        strategy_name="ema",
        parameter_space={},
        config=config
    )
    
    combinations = optimizer._generate_param_combinations()
    
    assert len(combinations) == 1, "Should return single empty dict"
    assert combinations[0] == {}, "Empty parameter space should return empty dict"
    
    print("✅ Espace de paramètres vide géré correctement")


def test_walk_forward_config_defaults():
    """Test valeurs par défaut de WalkForwardConfig"""
    config = WalkForwardConfig()
    
    assert config.train_period_days == 180
    assert config.test_period_days == 30
    assert config.anchored == False
    assert config.optimization_metric == "sharpe_ratio"
    assert config.min_trades == 10
    
    print("✅ Valeurs par défaut de WalkForwardConfig correctes")


def test_walk_forward_config_custom():
    """Test valeurs personnalisées de WalkForwardConfig"""
    config = WalkForwardConfig(
        train_period_days=90,
        test_period_days=15,
        anchored=True,
        optimization_metric="profit_factor",
        min_trades=5
    )
    
    assert config.train_period_days == 90
    assert config.test_period_days == 15
    assert config.anchored == True
    assert config.optimization_metric == "profit_factor"
    assert config.min_trades == 5
    
    print("✅ Valeurs personnalisées de WalkForwardConfig correctes")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTS UNITAIRES - WALK-FORWARD OPTIMIZATION")
    print("="*60 + "\n")
    
    test_window_generation_rolling()
    test_window_generation_anchored()
    test_parameter_combinations()
    test_empty_parameter_space()
    test_walk_forward_config_defaults()
    test_walk_forward_config_custom()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("="*60 + "\n")
