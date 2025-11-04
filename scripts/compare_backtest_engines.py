"""
Script de comparaison entre Backtrader (nouveau) et VectorBT (ancien)
"""
from datetime import datetime, timedelta
import pandas as pd
from src.backtesting.advanced_engine import AdvancedBacktestEngine, BacktestConfig, TransactionCosts
from src.backtesting.metrics import AdvancedMetrics
from src.backtesting.monte_carlo import MonteCarloSimulator


def compare_engines():
    """Comparer les moteurs de backtesting"""
    
    print("\n" + "="*80)
    print("🔬 COMPARAISON DES MOTEURS DE BACKTESTING")
    print("="*80)
    print("Backtrader (nouveau) vs VectorBT (ancien)")
    print("="*80 + "\n")
    
    # Configuration commune
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()
    
    # Test avec Backtrader (nouveau système)
    print("="*80)
    print("📊 TEST AVEC BACKTRADER (Nouveau Système)")
    print("="*80 + "\n")
    
    try:
        config = BacktestConfig(
            strategy_name='ema',
            strategy_params={'fast_period': 10, 'slow_period': 50},
            symbols=['BTC/USDT'],
            timeframe='1h',
            start_date=start,
            end_date=end,
            initial_capital=10000.0
        )
        
        engine = AdvancedBacktestEngine(config, TransactionCosts())
        metrics = engine.run()
        
        print("\n" + "="*80)
        print("📈 RÉSULTATS BACKTRADER")
        print("="*80)
        for key, value in metrics.items():
            if isinstance(value, (list, dict)):
                continue
            if isinstance(value, float):
                print(f"{key:.<40} {value:>15.4f}")
            else:
                print(f"{key:.<40} {str(value):>15}")
        
        # Essayer de générer le graphique
        try:
            print("\n📊 Génération du graphique...")
            engine.plot()
        except Exception as e:
            print(f"   ⚠️  Graphique non disponible (environnement sans display): {e}")
        
    except Exception as e:
        print(f"❌ Erreur lors du test Backtrader: {e}")
        import traceback
        traceback.print_exc()
    
    # Test avec VectorBT (ancien système)
    print("\n" + "="*80)
    print("📊 TEST AVEC VECTORBT (Ancien Système)")
    print("="*80)
    print("⚠️  Note: VectorBT nécessite implémentation supplémentaire")
    print("   Le système actuel se concentre sur Backtrader")
    print("="*80 + "\n")
    
    # Comparaison finale
    print("="*80)
    print("🏆 ANALYSE COMPARATIVE")
    print("="*80)
    print("\n✅ AVANTAGES DE BACKTRADER:")
    print("   • Transaction costs réalistes (maker/taker fees)")
    print("   • Slippage modeling inclus")
    print("   • Analyseurs institutionnels intégrés")
    print("   • Support multi-timeframe natif")
    print("   • Walk-forward optimization")
    print("   • Extensibilité via custom analyzers")
    
    print("\n📊 MÉTRIQUES DISPONIBLES:")
    print("   • Sharpe Ratio, Sortino Ratio, Calmar Ratio")
    print("   • Max Drawdown, Average Drawdown")
    print("   • Win Rate, Profit Factor")
    print("   • System Quality Number (SQN)")
    print("   • Variability-Weighted Return (VWR)")
    
    print("\n" + "="*80)


def demo_advanced_features():
    """Démonstration des fonctionnalités avancées"""
    
    print("\n" + "="*80)
    print("🚀 DÉMONSTRATION DES FONCTIONNALITÉS AVANCÉES")
    print("="*80 + "\n")
    
    # 1. Monte Carlo Simulation
    print("1️⃣  SIMULATION MONTE CARLO")
    print("-"*80)
    
    # Générer des trades fictifs pour démo
    import numpy as np
    np.random.seed(42)
    trades = np.random.normal(0.01, 0.05, 100)  # 100 trades avec moyenne 1%, std 5%
    
    try:
        simulator = MonteCarloSimulator(trades, n_simulations=1000)
        simulator.simulate()
        
        print("✅ Simulation Monte Carlo créée (1000 runs)")
        
        # Métriques de risque
        metrics = simulator.calculate_risk_metrics()
        print(f"\n📊 Métriques de risque:")
        print(f"   Rendement moyen:         {metrics['mean_return']:.2f}")
        print(f"   Probabilité de profit:   {metrics['prob_profit']:.1f}%")
        print(f"   Pire cas (5%):           {metrics['worst_case_5pct']:.2f}")
        print(f"   Drawdown moyen:          {metrics['mean_max_drawdown']:.2f}")
        
        # Générer rapport
        report = simulator.generate_risk_report(save_path="reports/monte_carlo_demo.txt")
        
        # Générer graphiques
        simulator.plot_simulations(save_path="reports/monte_carlo_demo.png")
        
    except Exception as e:
        print(f"⚠️  Erreur Monte Carlo: {e}")
    
    # 2. Métriques avancées
    print("\n2️⃣  MÉTRIQUES AVANCÉES (QuantStats)")
    print("-"*80)
    
    try:
        # Générer série de rendements fictifs
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        returns.index = pd.date_range(start='2023-01-01', periods=252, freq='D')
        
        metrics = AdvancedMetrics.calculate_all(returns)
        
        print("✅ Métriques calculées")
        print(f"\n📈 Exemples de métriques:")
        print(f"   Sharpe Ratio:            {metrics.get('sharpe', 0):.2f}")
        print(f"   Sortino Ratio:           {metrics.get('sortino', 0):.2f}")
        print(f"   Max Drawdown:            {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"   Win Rate:                {metrics.get('win_rate', 0):.1f}%")
        
        # Générer rapport
        report = AdvancedMetrics.generate_metrics_report(returns, output_path="reports/metrics_demo.txt")
        print(f"\n✅ Rapport sauvegardé: reports/metrics_demo.txt")
        
    except Exception as e:
        print(f"⚠️  Erreur métriques: {e}")
    
    # 3. Comparaison de stratégies
    print("\n3️⃣  COMPARAISON DE STRATÉGIES")
    print("-"*80)
    
    try:
        # Générer rendements pour 3 stratégies fictives
        strategies = {
            'Strategy A': pd.Series(np.random.normal(0.002, 0.015, 252)),
            'Strategy B': pd.Series(np.random.normal(0.001, 0.020, 252)),
            'Strategy C': pd.Series(np.random.normal(0.003, 0.025, 252)),
        }
        
        comparison_df = AdvancedMetrics.compare_strategies(strategies)
        
        print("✅ Comparaison effectuée\n")
        print(comparison_df.to_string())
        
    except Exception as e:
        print(f"⚠️  Erreur comparaison: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Comparaison principale
    compare_engines()
    
    # Démonstration fonctionnalités
    demo_advanced_features()
    
    print("\n" + "="*80)
    print("✅ TESTS TERMINÉS")
    print("="*80)
    print("\nFichiers générés:")
    print("   📄 reports/monte_carlo_demo.txt")
    print("   📊 reports/monte_carlo_demo.png")
    print("   📄 reports/metrics_demo.txt")
    print("\nProchaines étapes:")
    print("   1. Tester avec des données réelles")
    print("   2. Exécuter walk-forward optimization")
    print("   3. Analyser les résultats avec le dashboard Streamlit")
    print("="*80 + "\n")
