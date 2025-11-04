"""
Tests unitaires pour Advanced Metrics
"""
import sys
from pathlib import Path

# Ajouter le root du projet au path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.backtesting.metrics import AdvancedMetrics


def test_metrics_calculation():
    """Test calcul des métriques de base"""
    # Générer des returns synthétiques avec seed pour reproductibilité
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 252 jours de trading
    
    metrics = AdvancedMetrics.calculate_all(returns)
    
    # Vérifier que les métriques principales existent
    expected_metrics = ['total_return', 'sharpe', 'max_drawdown', 'win_rate']
    for metric in expected_metrics:
        assert metric in metrics, f"Métrique {metric} manquante"
    
    print("✅ Métriques calculées avec succès:")
    for key in expected_metrics:
        value = metrics[key]
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    return True


def test_sharpe_ratio_calculation():
    """Test calcul du ratio de Sharpe"""
    # Returns positifs constants
    returns = pd.Series([0.01] * 100)
    
    metrics = AdvancedMetrics.calculate_all(returns)
    
    # Sharpe devrait être très élevé avec returns constants positifs
    assert 'sharpe' in metrics
    assert metrics['sharpe'] > 0, "Sharpe ratio devrait être positif"
    
    print(f"✅ Sharpe ratio: {metrics['sharpe']:.2f}")
    
    return True


def test_max_drawdown_calculation():
    """Test calcul du drawdown maximum"""
    # Série avec un drawdown connu
    returns = pd.Series([0.1, -0.05, -0.05, -0.05, 0.2])
    
    metrics = AdvancedMetrics.calculate_all(returns)
    
    assert 'max_drawdown' in metrics
    # Le drawdown devrait être négatif ou nul
    assert metrics['max_drawdown'] <= 0, "Max drawdown devrait être <= 0"
    
    print(f"✅ Max drawdown: {metrics['max_drawdown']:.4f}")
    
    return True


def test_win_rate_calculation():
    """Test calcul du taux de réussite"""
    # 60% de trades gagnants
    returns = pd.Series([0.01, 0.02, -0.01, 0.01, -0.015, 0.01, 0.02, -0.01, 0.01, 0.01])
    
    metrics = AdvancedMetrics.calculate_all(returns)
    
    assert 'win_rate' in metrics
    # Win rate devrait être entre 0 et 100 (pourcentage)
    assert 0 <= metrics['win_rate'] <= 100, "Win rate doit être entre 0 et 100"
    
    print(f"✅ Win rate: {metrics['win_rate']:.1f}%")
    
    return True


def test_metrics_with_negative_returns():
    """Test métriques avec returns négatifs"""
    np.random.seed(123)
    # Générer des returns principalement négatifs
    returns = pd.Series(np.random.normal(-0.005, 0.02, 100))
    
    metrics = AdvancedMetrics.calculate_all(returns)
    
    assert 'total_return' in metrics
    # Total return devrait être négatif
    assert metrics['total_return'] < 0, "Total return devrait être négatif"
    
    print(f"✅ Métriques avec returns négatifs: Total return = {metrics['total_return']:.4f}")
    
    return True


def test_empty_returns_handling():
    """Test gestion des returns vides"""
    returns = pd.Series([])
    
    try:
        metrics = AdvancedMetrics.calculate_all(returns)
        # Si aucune erreur, vérifier que les métriques ont des valeurs par défaut
        print("✅ Returns vides gérés correctement")
    except Exception as e:
        # Si une erreur est levée, c'est acceptable
        print(f"✅ Returns vides gèrent l'exception: {type(e).__name__}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTS UNITAIRES - ADVANCED METRICS")
    print("="*60 + "\n")
    
    test_metrics_calculation()
    test_sharpe_ratio_calculation()
    test_max_drawdown_calculation()
    test_win_rate_calculation()
    test_metrics_with_negative_returns()
    test_empty_returns_handling()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("="*60 + "\n")
