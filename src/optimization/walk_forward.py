"""
Walk-Forward Optimization pour éviter l'overfitting
"""
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import itertools
from src.backtesting.advanced_engine import AdvancedBacktestEngine, BacktestConfig, TransactionCosts


@dataclass
class WalkForwardConfig:
    """Configuration walk-forward"""
    train_period_days: int = 180  # 6 mois entraînement
    test_period_days: int = 30    # 1 mois test
    anchored: bool = False         # False = rolling window
    optimization_metric: str = "sharpe_ratio"  # Métrique à optimiser
    min_trades: int = 10          # Minimum de trades pour validation
    

class WalkForwardOptimizer:
    """
    Walk-forward optimization pour éviter l'overfitting
    
    Principe:
    1. Diviser les données en fenêtres train/test
    2. Optimiser paramètres sur train
    3. Tester sur test (out-of-sample)
    4. Répéter pour toutes les fenêtres
    """
    
    def __init__(self, 
                 strategy_name: str,
                 parameter_space: Dict[str, List[Any]],
                 config: WalkForwardConfig):
        self.strategy_name = strategy_name
        self.parameter_space = parameter_space
        self.config = config
        self.results = []
        
    def generate_windows(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Générer les fenêtres train/test
        
        Returns:
            List[(train_start, train_end, test_start, test_end)]
        """
        windows = []
        current_start = start_date
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.config.train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_period_days)
            
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Rolling window vs Anchored
            if not self.config.anchored:
                # Rolling: avance de test_period
                current_start = test_start
            else:
                # Anchored: garde le même point de départ
                current_start = start_date
                # Mais avance la fin de test
                if len(windows) > 0:
                    break  # Pour anchored simple
        
        return windows
    
    def optimize_parameters(self, 
                           train_start: datetime, 
                           train_end: datetime, 
                           symbols: List[str], 
                           timeframe: str) -> Tuple[Dict[str, Any], float]:
        """
        Optimiser les paramètres sur la période d'entraînement
        
        Returns:
            (best_params, best_score)
        """
        best_params = None
        best_score = -np.inf
        
        # Générer toutes les combinaisons de paramètres
        param_combinations = self._generate_param_combinations()
        
        print(f"🔍 Optimisation sur {len(param_combinations)} combinaisons...")
        
        valid_combinations = 0
        
        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                print(f"   Progression: {i+1}/{len(param_combinations)}")
            
            try:
                backtest_config = BacktestConfig(
                    strategy_name=self.strategy_name,
                    strategy_params=params,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=10000.0
                )
                
                engine = AdvancedBacktestEngine(backtest_config, TransactionCosts())
                metrics = engine.run()
                
                # Validation: minimum de trades
                if metrics.get('total_trades', 0) < self.config.min_trades:
                    continue
                
                valid_combinations += 1
                score = metrics.get(self.config.optimization_metric, -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                print(f"   ⚠️  Erreur avec params {params}: {e}")
                continue
        
        if best_params is None:
            print(f"   ⚠️  Aucune combinaison valide trouvée!")
            # Retourner première combinaison par défaut
            best_params = param_combinations[0] if param_combinations else {}
            best_score = -np.inf
        else:
            print(f"✅ Meilleurs paramètres: {best_params}")
            print(f"   Score ({self.config.optimization_metric}): {best_score:.4f}")
            print(f"   Combinaisons valides: {valid_combinations}/{len(param_combinations)}")
        
        return best_params, best_score
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Générer toutes les combinaisons de paramètres"""
        if not self.parameter_space:
            return [{}]
        
        keys = list(self.parameter_space.keys())
        values = [self.parameter_space[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def run_walk_forward(self, 
                        start_date: datetime, 
                        end_date: datetime, 
                        symbols: List[str], 
                        timeframe: str) -> pd.DataFrame:
        """
        Exécuter walk-forward optimization complète
        
        Returns:
            DataFrame avec résultats de chaque fenêtre
        """
        windows = self.generate_windows(start_date, end_date)
        
        print("\n" + "="*60)
        print("🔄 WALK-FORWARD OPTIMIZATION")
        print("="*60)
        print(f"📊 Stratégie: {self.strategy_name}")
        print(f"📈 Symboles: {', '.join(symbols)}")
        print(f"⏰ Période totale: {start_date.date()} → {end_date.date()}")
        print(f"🪟 Nombre de fenêtres: {len(windows)}")
        print(f"📅 Train: {self.config.train_period_days} jours")
        print(f"📅 Test: {self.config.test_period_days} jours")
        print(f"🔄 Type: {'Anchored' if self.config.anchored else 'Rolling'}")
        print(f"🎯 Métrique: {self.config.optimization_metric}")
        print("="*60 + "\n")
        
        results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"🔄 Fenêtre {i+1}/{len(windows)}")
            print(f"{'='*60}")
            print(f"📅 Train: {train_start.date()} → {train_end.date()} ({(train_end - train_start).days} jours)")
            print(f"📅 Test:  {test_start.date()} → {test_end.date()} ({(test_end - test_start).days} jours)")
            
            # Phase 1: Optimiser sur train
            print(f"\n🔍 PHASE 1: OPTIMISATION (in-sample)")
            best_params, train_score = self.optimize_parameters(
                train_start, train_end, symbols, timeframe
            )
            
            # Phase 2: Tester sur test (out-of-sample)
            print(f"\n🧪 PHASE 2: TEST (out-of-sample)")
            try:
                backtest_config = BacktestConfig(
                    strategy_name=self.strategy_name,
                    strategy_params=best_params,
                    symbols=symbols,
                    timeframe=timeframe,
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=10000.0
                )
                
                engine = AdvancedBacktestEngine(backtest_config, TransactionCosts())
                test_metrics = engine.run()
                
                # Calculer dégradation
                test_score = test_metrics.get(self.config.optimization_metric, 0)
                degradation = ((test_score - train_score) / abs(train_score) * 100) if train_score != 0 else 0
                
                print(f"\n📊 Comparaison In-Sample vs Out-of-Sample:")
                print(f"   Train ({self.config.optimization_metric}): {train_score:.4f}")
                print(f"   Test ({self.config.optimization_metric}): {test_score:.4f}")
                print(f"   Dégradation: {degradation:+.2f}%")
                
                results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_params': str(best_params),
                    'train_score': train_score,
                    'test_score': test_score,
                    'degradation_pct': degradation,
                    **{f'test_{k}': v for k, v in test_metrics.items()}
                })
                
            except Exception as e:
                print(f"⚠️  Erreur lors du test: {e}")
                results.append({
                    'window': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_params': str(best_params),
                    'train_score': train_score,
                    'test_score': np.nan,
                    'degradation_pct': np.nan,
                    'error': str(e)
                })
        
        # Créer DataFrame des résultats
        df_results = pd.DataFrame(results)
        
        # Résumé final
        print("\n" + "="*60)
        print("📊 RÉSUMÉ WALK-FORWARD")
        print("="*60)
        
        if len(df_results) > 0:
            valid_windows = df_results[df_results['test_score'].notna()]
            
            if len(valid_windows) > 0:
                print(f"✅ Fenêtres valides: {len(valid_windows)}/{len(df_results)}")
                print(f"\n📈 Statistiques Test (out-of-sample):")
                print(f"   {self.config.optimization_metric} moyen: {valid_windows['test_score'].mean():.4f}")
                print(f"   {self.config.optimization_metric} médian: {valid_windows['test_score'].median():.4f}")
                print(f"   Dégradation moyenne: {valid_windows['degradation_pct'].mean():.2f}%")
                
                if 'test_win_rate' in valid_windows.columns:
                    print(f"   Win rate moyen: {valid_windows['test_win_rate'].mean():.2f}%")
                if 'test_profit_factor' in valid_windows.columns:
                    print(f"   Profit factor moyen: {valid_windows['test_profit_factor'].mean():.2f}")
            else:
                print("⚠️  Aucune fenêtre valide")
        
        print("="*60 + "\n")
        
        return df_results
    
    def analyze_stability(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyser la stabilité des résultats walk-forward
        
        Returns:
            Dict avec métriques de stabilité
        """
        valid_results = results_df[results_df['test_score'].notna()]
        
        if len(valid_results) == 0:
            return {
                'stability_score': 0,
                'consistency_rate': 0,
                'avg_degradation': 0
            }
        
        # Taux de cohérence (fenêtres profitables)
        profitable_windows = len(valid_results[valid_results['test_pnl'] > 0])
        consistency_rate = (profitable_windows / len(valid_results)) * 100
        
        # Stabilité du score (inverse de la volatilité)
        score_std = valid_results['test_score'].std()
        score_mean = valid_results['test_score'].mean()
        stability_score = (score_mean / score_std) if score_std > 0 else 0
        
        # Dégradation moyenne
        avg_degradation = valid_results['degradation_pct'].mean()
        
        return {
            'stability_score': stability_score,
            'consistency_rate': consistency_rate,
            'avg_degradation': avg_degradation,
            'score_mean': score_mean,
            'score_std': score_std,
            'profitable_windows': profitable_windows,
            'total_windows': len(valid_results)
        }
