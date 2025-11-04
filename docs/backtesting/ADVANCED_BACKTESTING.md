# Système de Backtesting Avancé

## Vue d'Ensemble

Le système de backtesting avancé implémente des fonctionnalités de niveau institutionnel pour l'analyse de stratégies de trading, incluant:

- **Moteur Backtrader** avec transaction costs réalistes
- **Walk-Forward Optimization** pour éviter l'overfitting
- **Métriques avancées** (QuantStats integration)
- **Simulation Monte Carlo** pour analyse de risque
- **Transaction Cost Analysis** (TCA) avec maker/taker fees

---

## Architecture

```
src/backtesting/
├── __init__.py
├── advanced_engine.py      # Moteur principal Backtrader
├── metrics.py              # Métriques institutionnelles
└── monte_carlo.py          # Simulation Monte Carlo

src/optimization/
├── __init__.py
└── walk_forward.py         # Walk-forward optimization
```

---

## 1. Advanced Backtest Engine

### Fonctionnalités

- **Transaction Costs Réalistes**
  - Maker fee: 0.02% (Binance)
  - Taker fee: 0.04% (Binance)
  - Modèles de slippage: fixed, volumetric, sqrt

- **Analyseurs Intégrés**
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Max Drawdown
  - System Quality Number (SQN)
  - Variability-Weighted Return (VWR)

### Utilisation

```python
from datetime import datetime, timedelta
from src.backtesting.advanced_engine import (
    AdvancedBacktestEngine, 
    BacktestConfig, 
    TransactionCosts
)

# Configuration
config = BacktestConfig(
    strategy_name='ema',
    strategy_params={'fast_period': 10, 'slow_period': 50},
    symbols=['BTC/USDT'],
    timeframe='1h',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    initial_capital=10000.0
)

# Transaction costs
costs = TransactionCosts(
    maker_fee=0.0002,  # 0.02%
    taker_fee=0.0004,  # 0.04%
    slippage_model='fixed',
    slippage_basis_points=5
)

# Exécuter backtest
engine = AdvancedBacktestEngine(config, costs)
metrics = engine.run()

# Afficher résultats
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### Métriques Retournées

```python
{
    'initial_capital': 10000.0,
    'final_value': 12500.0,
    'pnl': 2500.0,
    'pnl_pct': 25.0,
    'sharpe_ratio': 1.85,
    'max_drawdown': -12.5,
    'total_trades': 45,
    'won_trades': 28,
    'lost_trades': 17,
    'win_rate': 62.22,
    'profit_factor': 1.75,
    'sqn': 2.1
}
```

---

## 2. Walk-Forward Optimization

### Principe

Le walk-forward évite l'overfitting en:
1. Divisant les données en fenêtres train/test
2. Optimisant les paramètres sur train
3. Testant sur test (out-of-sample)
4. Répétant pour toutes les fenêtres

### Types de Windows

**Rolling Window** (recommandé):
```
Train: [0...180]    Test: [180...210]
Train: [30...210]   Test: [210...240]
Train: [60...240]   Test: [240...270]
```

**Anchored Window**:
```
Train: [0...180]    Test: [180...210]
Train: [0...210]    Test: [210...240]
Train: [0...240]    Test: [240...270]
```

### Utilisation

```python
from datetime import datetime
from src.optimization.walk_forward import (
    WalkForwardOptimizer,
    WalkForwardConfig
)

# Configuration
config = WalkForwardConfig(
    train_period_days=180,  # 6 mois
    test_period_days=30,    # 1 mois
    anchored=False,         # Rolling window
    optimization_metric='sharpe_ratio',
    min_trades=10
)

# Espace de paramètres à optimiser
parameter_space = {
    'fast_period': [5, 10, 15, 20],
    'slow_period': [30, 50, 100, 200]
}

# Créer optimizer
optimizer = WalkForwardOptimizer(
    strategy_name='ema',
    parameter_space=parameter_space,
    config=config
)

# Exécuter walk-forward
results_df = optimizer.run_walk_forward(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    symbols=['BTC/USDT'],
    timeframe='1h'
)

# Analyser stabilité
stability = optimizer.analyze_stability(results_df)
print(f"Consistency Rate: {stability['consistency_rate']:.1f}%")
print(f"Average Degradation: {stability['avg_degradation']:.1f}%")
```

### Interpréter les Résultats

- **Dégradation < 20%**: Stratégie robuste
- **Dégradation 20-50%**: Overfitting modéré
- **Dégradation > 50%**: Overfitting sévère

---

## 3. Métriques Avancées

### Catégories de Métriques

#### Performance
- **Total Return**: Rendement total
- **CAGR**: Compound Annual Growth Rate
- **Average Return**: Rendement moyen
- **Best/Worst Day**: Meilleur/Pire jour

#### Risque
- **Volatility**: Volatilité (annualisée)
- **Max Drawdown**: Drawdown maximum
- **VaR 95%**: Value at Risk à 95%
- **CVaR 95%**: Conditional VaR à 95%

#### Ratios Ajustés au Risque
- **Sharpe Ratio**: (Return - RiskFree) / Volatility
- **Sortino Ratio**: Return / Downside Deviation
- **Calmar Ratio**: CAGR / Max Drawdown

#### Trading
- **Win Rate**: Taux de trades gagnants
- **Profit Factor**: Gains / Pertes
- **Payoff Ratio**: Gain moyen / Perte moyenne

### Utilisation

```python
from src.backtesting.metrics import AdvancedMetrics
import pandas as pd

# Calcul returns series depuis backtest
# (à implémenter selon votre backtest)
returns = pd.Series([...])  # Daily returns

# Calculer toutes les métriques
metrics = AdvancedMetrics.calculate_all(
    returns, 
    benchmark_returns=None,  # Optionnel
    risk_free_rate=0.0
)

# Générer rapport texte
report = AdvancedMetrics.generate_metrics_report(
    returns,
    output_path='reports/strategy_metrics.txt'
)
print(report)

# Générer tearsheet HTML (nécessite QuantStats)
AdvancedMetrics.generate_tearsheet(
    returns,
    output_path='reports/strategy_tearsheet.html',
    title='Ma Stratégie'
)

# Comparer stratégies
strategies = {
    'Strategy A': returns_a,
    'Strategy B': returns_b,
    'Strategy C': returns_c
}
comparison = AdvancedMetrics.compare_strategies(strategies)
print(comparison)
```

---

## 4. Simulation Monte Carlo

### Principe

Utilise bootstrap sampling sur les trades historiques pour:
- Estimer distribution des rendements futurs
- Calculer probabilités de profit/perte
- Évaluer risque de drawdown extrême

### Utilisation

```python
from src.backtesting.monte_carlo import MonteCarloSimulator

# Liste des P&L de chaque trade
trades = [100, -50, 75, -25, 150, ...]  # De votre backtest

# Créer simulator
simulator = MonteCarloSimulator(
    trades=trades,
    n_simulations=10000
)

# Exécuter simulation
simulations = simulator.simulate(n_trades=len(trades))

# Métriques de risque
metrics = simulator.calculate_risk_metrics()
print(f"Probabilité de profit: {metrics['prob_profit']:.1f}%")
print(f"Pire cas (5%): {metrics['worst_case_5pct']:.2f}")
print(f"Prob DD > 20%: {metrics['prob_drawdown_20pct']:.1f}%")

# Générer rapport
report = simulator.generate_risk_report(
    save_path='reports/monte_carlo.txt'
)

# Visualiser
simulator.plot_simulations(
    save_path='reports/monte_carlo.png'
)

# Stress test
stress = simulator.stress_test(worst_case_percentile=5)
print(f"Perte moyenne pire 5%: {stress['avg_loss_in_worst_case']:.2f}")
```

---

## 5. Transaction Cost Analysis (TCA)

### Modèles de Slippage

#### Fixed Slippage
```python
slippage = price * (basis_points / 10000)
```

#### Volumetric Slippage
```python
slippage = price * (basis_points / 10000) * (volume / 1000)
```

#### Square Root Slippage
```python
slippage = price * (basis_points / 10000) * sqrt(volume / 1000)
```

### Configuration

```python
from src.backtesting.advanced_engine import TransactionCosts

# Conservative (haute liquidité)
costs_conservative = TransactionCosts(
    maker_fee=0.0001,  # 0.01%
    taker_fee=0.0002,  # 0.02%
    slippage_model='fixed',
    slippage_basis_points=2
)

# Realistic (liquidité moyenne)
costs_realistic = TransactionCosts(
    maker_fee=0.0002,  # 0.02%
    taker_fee=0.0004,  # 0.04%
    slippage_model='sqrt',
    slippage_basis_points=5
)

# Pessimistic (faible liquidité)
costs_pessimistic = TransactionCosts(
    maker_fee=0.0005,  # 0.05%
    taker_fee=0.001,   # 0.10%
    slippage_model='volumetric',
    slippage_basis_points=10
)
```

---

## 6. Best Practices

### Éviter l'Overfitting

1. **Utiliser Walk-Forward**: Toujours valider out-of-sample
2. **Limiter l'Espace de Paramètres**: Moins de paramètres = moins d'overfitting
3. **Cross-Validation**: Tester sur plusieurs périodes
4. **Minimum de Trades**: Au moins 30-50 trades par test

### Analyse de Robustesse

```python
# 1. Walk-forward avec différentes fenêtres
configs = [
    WalkForwardConfig(train_period_days=90, test_period_days=15),
    WalkForwardConfig(train_period_days=180, test_period_days=30),
    WalkForwardConfig(train_period_days=365, test_period_days=60),
]

# 2. Tester plusieurs symboles
symbols_sets = [
    ['BTC/USDT'],
    ['ETH/USDT'],
    ['BTC/USDT', 'ETH/USDT']
]

# 3. Différents timeframes
timeframes = ['1h', '4h', '1d']

# 4. Scénarios de coûts
cost_scenarios = [
    TransactionCosts(taker_fee=0.0002),  # Optimiste
    TransactionCosts(taker_fee=0.0004),  # Réaliste
    TransactionCosts(taker_fee=0.0006),  # Pessimiste
]
```

### Métriques Critiques

Pour qu'une stratégie soit viable:
- **Sharpe Ratio > 1.0** (idéalement > 1.5)
- **Win Rate > 45%** (ou Profit Factor > 1.5)
- **Max Drawdown < 25%**
- **Cohérence Walk-Forward > 60%**
- **Dégradation < 30%**

---

## 7. Workflow Complet

```python
from datetime import datetime
from src.backtesting.advanced_engine import (
    AdvancedBacktestEngine, BacktestConfig, TransactionCosts
)
from src.optimization.walk_forward import (
    WalkForwardOptimizer, WalkForwardConfig
)
from src.backtesting.metrics import AdvancedMetrics
from src.backtesting.monte_carlo import MonteCarloSimulator

# 1. Configuration
start = datetime(2023, 1, 1)
end = datetime(2024, 1, 1)

# 2. Walk-Forward Optimization
wf_config = WalkForwardConfig(
    train_period_days=180,
    test_period_days=30,
    optimization_metric='sharpe_ratio'
)

optimizer = WalkForwardOptimizer(
    strategy_name='ema',
    parameter_space={
        'fast_period': [10, 15, 20],
        'slow_period': [50, 100, 200]
    },
    config=wf_config
)

results_df = optimizer.run_walk_forward(
    start_date=start,
    end_date=end,
    symbols=['BTC/USDT'],
    timeframe='1h'
)

# 3. Analyser stabilité
stability = optimizer.analyze_stability(results_df)
print(f"\nStability Score: {stability['stability_score']:.2f}")
print(f"Consistency Rate: {stability['consistency_rate']:.1f}%")

# 4. Backtest final avec meilleurs paramètres
best_params = eval(results_df.iloc[0]['best_params'])

config = BacktestConfig(
    strategy_name='ema',
    strategy_params=best_params,
    symbols=['BTC/USDT'],
    timeframe='1h',
    start_date=start,
    end_date=end
)

engine = AdvancedBacktestEngine(config, TransactionCosts())
final_metrics = engine.run()

# 5. Analyse Monte Carlo (si trades disponibles)
# trades = engine.get_trades_list()
# simulator = MonteCarloSimulator(trades, n_simulations=10000)
# mc_metrics = simulator.calculate_risk_metrics()

# 6. Rapport final
print("\n" + "="*60)
print("📊 RAPPORT FINAL")
print("="*60)
print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {final_metrics['max_drawdown']:.2f}%")
print(f"Win Rate: {final_metrics['win_rate']:.1f}%")
print(f"Profit Factor: {final_metrics['profit_factor']:.2f}")
```

---

## 8. Troubleshooting

### Problème: Pas de trades générés

**Solution**:
- Vérifier que les données sont chargées
- Vérifier les paramètres de la stratégie
- Réduire le `min_trades` dans `WalkForwardConfig`

### Problème: Métriques NaN

**Solution**:
- Vérifier qu'il y a assez de données
- S'assurer que les returns sont calculés correctement
- Installer QuantStats: `pip install quantstats`

### Problème: Walk-forward très lent

**Solution**:
- Réduire l'espace de paramètres
- Utiliser des périodes plus courtes
- Paralléliser (future feature)

---

## 9. Références

- [Backtrader Documentation](https://www.backtrader.com/docu/)
- [QuantStats Documentation](https://github.com/ranaroussi/quantstats)
- [Walk-Forward Analysis](https://www.investopedia.com/terms/w/walk-forward-analysis.asp)
- [Monte Carlo Methods in Trading](https://www.quantstart.com/articles/monte-carlo-methods-in-quantitative-finance/)

---

## 10. Prochaines Étapes

- [ ] Dashboard Streamlit interactif
- [ ] Parallélisation walk-forward
- [ ] Support multi-asset portfolio
- [ ] Machine learning optimization
- [ ] Real-time monitoring
