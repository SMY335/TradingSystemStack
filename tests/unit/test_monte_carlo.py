"""
Tests unitaires pour Monte Carlo Simulator
"""
import sys
from pathlib import Path

# Ajouter le root du projet au path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.backtesting.monte_carlo import MonteCarloSimulator


def test_monte_carlo_simulation():
    """Test simulation Monte Carlo de base"""
    # Trades synthétiques avec un mix de gains et pertes
    trades = [0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012, -0.003, 0.018, -0.007]
    
    simulator = MonteCarloSimulator(trades, n_simulations=1000)
    simulations = simulator.simulate(n_trades=50)
    
    # Vérifier la forme du résultat
    assert simulations.shape == (1000, 50), f"Expected shape (1000, 50), got {simulations.shape}"
    
    risk_metrics = simulator.calculate_risk_metrics(simulations)
    
    # Vérifier les métriques de risque
    expected_keys = ['mean_return', 'prob_profit', 'worst_case_5pct', 'best_case_95pct']
    for key in expected_keys:
        assert key in risk_metrics, f"Métrique {key} manquante"
    
    print("✅ Monte Carlo simulation réussie:")
    print(f"   Simulations: {simulations.shape[0]} × {simulations.shape[1]} trades")
    print(f"   Mean return: {risk_metrics['mean_return']:.4f}")
    print(f"   Prob profit: {risk_metrics['prob_profit']:.2f}%")
    
    return True


def test_probability_of_profit():
    """Test calcul de probabilité de profit"""
    # Trades majoritairement positifs
    trades = [0.02, 0.01, 0.015, -0.005, 0.01, 0.018, 0.012, -0.003, 0.02, 0.015]
    
    simulator = MonteCarloSimulator(trades, n_simulations=500)
    simulations = simulator.simulate(n_trades=30)
    
    risk_metrics = simulator.calculate_risk_metrics(simulations)
    
    # Avec des trades majoritairement positifs, prob_profit devrait être élevée
    assert risk_metrics['prob_profit'] > 50, "Probabilité de profit devrait être > 50%"
    
    print(f"✅ Probabilité de profit: {risk_metrics['prob_profit']:.1f}%")
    
    return True


def test_worst_case_scenarios():
    """Test calcul des scénarios worst-case"""
    trades = [0.01, -0.02, 0.015, -0.01, 0.02, -0.015]
    
    simulator = MonteCarloSimulator(trades, n_simulations=1000)
    simulations = simulator.simulate(n_trades=40)
    
    risk_metrics = simulator.calculate_risk_metrics(simulations)
    
    # Vérifier les percentiles worst-case
    assert 'worst_case_5pct' in risk_metrics
    assert 'worst_case_10pct' in risk_metrics
    
    # Worst case 5% devrait être pire que worst case 10%
    assert risk_metrics['worst_case_5pct'] <= risk_metrics['worst_case_10pct'], \
        "Worst case 5% devrait être <= worst case 10%"
    
    print("✅ Métriques de risque:")
    print(f"   Worst case 5%: {risk_metrics['worst_case_5pct']:.4f}")
    print(f"   Worst case 10%: {risk_metrics['worst_case_10pct']:.4f}")
    
    return True


def test_simulation_reproducibility():
    """Test reproductibilité des simulations"""
    trades = [0.01, -0.005, 0.02, -0.01, 0.015]
    
    # Première simulation
    np.random.seed(42)
    simulator1 = MonteCarloSimulator(trades, n_simulations=100)
    sim1 = simulator1.simulate(n_trades=20)
    
    # Deuxième simulation avec même seed
    np.random.seed(42)
    simulator2 = MonteCarloSimulator(trades, n_simulations=100)
    sim2 = simulator2.simulate(n_trades=20)
    
    # Les résultats devraient être identiques
    assert np.allclose(sim1, sim2), "Les simulations avec même seed devraient être identiques"
    
    print("✅ Reproductibilité des simulations vérifiée")
    
    return True


def test_different_simulation_sizes():
    """Test avec différentes tailles de simulation"""
    trades = [0.01, -0.005, 0.02]
    
    for n_sims in [100, 500, 1000]:
        for n_trades in [10, 25, 50]:
            simulator = MonteCarloSimulator(trades, n_simulations=n_sims)
            simulations = simulator.simulate(n_trades=n_trades)
            
            assert simulations.shape == (n_sims, n_trades)
    
    print("✅ Différentes tailles de simulation testées")
    
    return True


def test_negative_trades():
    """Test avec trades majoritairement négatifs"""
    # Trades perdants
    trades = [-0.02, -0.01, 0.005, -0.015, -0.01, 0.003, -0.02]
    
    simulator = MonteCarloSimulator(trades, n_simulations=500)
    simulations = simulator.simulate(n_trades=30)
    
    risk_metrics = simulator.calculate_risk_metrics(simulations)
    
    # Mean return devrait être négatif
    assert risk_metrics['mean_return'] < 0, "Mean return devrait être négatif"
    
    # Probabilité de profit devrait être faible
    assert risk_metrics['prob_profit'] < 50, "Probabilité de profit devrait être < 50%"
    
    print("✅ Trades négatifs gérés correctement:")
    print(f"   Mean return: {risk_metrics['mean_return']:.4f}")
    print(f"   Prob profit: {risk_metrics['prob_profit']:.1f}%")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTS UNITAIRES - MONTE CARLO SIMULATOR")
    print("="*60 + "\n")
    
    test_monte_carlo_simulation()
    test_probability_of_profit()
    test_worst_case_scenarios()
    test_simulation_reproducibility()
    test_different_simulation_sizes()
    test_negative_trades()
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("="*60 + "\n")
