"""
Système de métriques avancées de niveau institutionnel avec QuantStats
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings

# Essayer d'importer quantstats, avec fallback
try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    warnings.warn("QuantStats not installed. Some advanced metrics will be unavailable.")


class AdvancedMetrics:
    """
    Métriques de niveau institutionnel pour l'analyse de stratégies
    
    Inclut:
    - Rendements et performance
    - Métriques de risque (volatilité, drawdown, VaR, CVaR)
    - Ratios ajustés au risque (Sharpe, Sortino, Calmar)
    - Métriques de trading (win rate, profit factor)
    - Comparaison au benchmark
    """
    
    @staticmethod
    def calculate_all(returns: pd.Series, 
                     benchmark_returns: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Calculer toutes les métriques disponibles
        
        Args:
            returns: Series des rendements (daily returns)
            benchmark_returns: Series des rendements du benchmark (optional)
            risk_free_rate: Taux sans risque annualisé (default: 0)
            
        Returns:
            Dict contenant toutes les métriques
        """
        if not HAS_QUANTSTATS:
            return AdvancedMetrics._calculate_basic_metrics(returns)
        
        metrics = {}
        
        # S'assurer que returns est une Series avec index datetime
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        try:
            # === RENDEMENTS ===
            metrics['total_return'] = qs.stats.comp(returns)
            metrics['cagr'] = qs.stats.cagr(returns)
            metrics['avg_return'] = qs.stats.avg_return(returns)
            metrics['avg_win'] = qs.stats.avg_win(returns)
            metrics['avg_loss'] = qs.stats.avg_loss(returns)
            metrics['best_day'] = qs.stats.best(returns)
            metrics['worst_day'] = qs.stats.worst(returns)
            
            # === RISQUE ===
            metrics['volatility'] = qs.stats.volatility(returns)
            metrics['volatility_annual'] = qs.stats.volatility(returns, periods=252)
            
            # Ratios ajustés au risque
            metrics['sharpe'] = qs.stats.sharpe(returns, rf=risk_free_rate)
            metrics['sortino'] = qs.stats.sortino(returns, rf=risk_free_rate)
            metrics['calmar'] = qs.stats.calmar(returns)
            
            # Drawdown
            metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
            metrics['avg_drawdown'] = qs.stats.avg_drawdown(returns)
            metrics['avg_drawdown_days'] = qs.stats.avg_drawdown_days(returns)
            
            # === TRADING ===
            metrics['win_rate'] = qs.stats.win_rate(returns)
            metrics['profit_factor'] = qs.stats.profit_factor(returns)
            metrics['payoff_ratio'] = qs.stats.payoff_ratio(returns)
            metrics['win_loss_ratio'] = qs.stats.win_loss_ratio(returns)
            
            # === RISQUE EXTRÊME ===
            metrics['var_95'] = qs.stats.value_at_risk(returns, sigma=1.65)  # 95% VaR
            metrics['cvar_95'] = qs.stats.conditional_value_at_risk(returns, sigma=1.65)  # 95% CVaR
            metrics['var_99'] = qs.stats.value_at_risk(returns, sigma=2.33)  # 99% VaR
            metrics['tail_ratio'] = qs.stats.tail_ratio(returns)
            
            # === STABILITÉ ===
            metrics['recovery_factor'] = qs.stats.recovery_factor(returns)
            metrics['ulcer_index'] = qs.stats.ulcer_index(returns)
            metrics['serenity_index'] = qs.stats.ulcer_performance_index(returns)
            
            # === BENCHMARK COMPARISON ===
            if benchmark_returns is not None:
                try:
                    metrics['alpha'] = qs.stats.alpha(returns, benchmark_returns, rf=risk_free_rate)
                    metrics['beta'] = qs.stats.beta(returns, benchmark_returns)
                    metrics['information_ratio'] = qs.stats.information_ratio(returns, benchmark_returns)
                    metrics['r_squared'] = qs.stats.r_squared(returns, benchmark_returns)
                    metrics['tracking_error'] = qs.stats.volatility(returns - benchmark_returns)
                except Exception as e:
                    print(f"⚠️  Erreur calcul métriques benchmark: {e}")
            
            # === QUALITÉ DES RENDEMENTS ===
            metrics['skew'] = qs.stats.skew(returns)
            metrics['kurtosis'] = qs.stats.kurtosis(returns)
            metrics['kelly_criterion'] = qs.stats.kelly_criterion(returns)
            
            # Nettoyer NaN/Inf
            for key, value in metrics.items():
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        metrics[key] = 0.0
            
        except Exception as e:
            print(f"⚠️  Erreur lors du calcul des métriques: {e}")
            metrics.update(AdvancedMetrics._calculate_basic_metrics(returns))
        
        return metrics
    
    @staticmethod
    def _calculate_basic_metrics(returns: pd.Series) -> Dict[str, Any]:
        """Métriques de base sans QuantStats"""
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        
        total_return = (1 + returns).prod() - 1
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Sharpe ratio simple
        sharpe = (avg_return / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = (winning_days / total_days * 100) if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
        }
    
    @staticmethod
    def generate_tearsheet(returns: pd.Series, 
                          benchmark: Optional[pd.Series] = None,
                          output_path: str = "reports/tearsheet.html",
                          title: str = "Strategy Performance"):
        """
        Générer un rapport complet HTML avec QuantStats
        
        Args:
            returns: Series des rendements
            benchmark: Series du benchmark (optional)
            output_path: Chemin de sortie du fichier HTML
            title: Titre du rapport
        """
        if not HAS_QUANTSTATS:
            print("⚠️  QuantStats non installé. Impossible de générer le tearsheet.")
            print("   Installation: pip install quantstats")
            return
        
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if benchmark is not None:
                qs.reports.html(returns, benchmark=benchmark, output=output_path, title=title)
            else:
                qs.reports.html(returns, output=output_path, title=title)
            
            print(f"✅ Tearsheet généré: {output_path}")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la génération du tearsheet: {e}")
    
    @staticmethod
    def generate_metrics_report(returns: pd.Series,
                               benchmark: Optional[pd.Series] = None,
                               output_path: str = "reports/metrics.txt") -> str:
        """
        Générer un rapport texte des métriques
        
        Returns:
            String contenant le rapport formaté
        """
        metrics = AdvancedMetrics.calculate_all(returns, benchmark)
        
        report = []
        report.append("=" * 60)
        report.append("📊 RAPPORT DE MÉTRIQUES AVANCÉES")
        report.append("=" * 60)
        
        # Performance
        report.append("\n📈 PERFORMANCE")
        report.append("-" * 60)
        report.append(f"Rendement Total:        {metrics.get('total_return', 0)*100:>10.2f}%")
        report.append(f"CAGR:                   {metrics.get('cagr', 0)*100:>10.2f}%")
        report.append(f"Rendement Moyen:        {metrics.get('avg_return', 0)*100:>10.2f}%")
        report.append(f"Meilleur Jour:          {metrics.get('best_day', 0)*100:>10.2f}%")
        report.append(f"Pire Jour:              {metrics.get('worst_day', 0)*100:>10.2f}%")
        
        # Risque
        report.append("\n⚠️  RISQUE")
        report.append("-" * 60)
        report.append(f"Volatilité:             {metrics.get('volatility', 0)*100:>10.2f}%")
        report.append(f"Volatilité Annuelle:    {metrics.get('volatility_annual', 0)*100:>10.2f}%")
        report.append(f"Max Drawdown:           {metrics.get('max_drawdown', 0)*100:>10.2f}%")
        report.append(f"Avg Drawdown:           {metrics.get('avg_drawdown', 0)*100:>10.2f}%")
        report.append(f"VaR 95%:                {metrics.get('var_95', 0)*100:>10.2f}%")
        report.append(f"CVaR 95%:               {metrics.get('cvar_95', 0)*100:>10.2f}%")
        
        # Ratios
        report.append("\n📊 RATIOS AJUSTÉS AU RISQUE")
        report.append("-" * 60)
        report.append(f"Sharpe Ratio:           {metrics.get('sharpe', 0):>10.2f}")
        report.append(f"Sortino Ratio:          {metrics.get('sortino', 0):>10.2f}")
        report.append(f"Calmar Ratio:           {metrics.get('calmar', 0):>10.2f}")
        
        # Trading
        report.append("\n💰 MÉTRIQUES DE TRADING")
        report.append("-" * 60)
        report.append(f"Win Rate:               {metrics.get('win_rate', 0):>10.2f}%")
        report.append(f"Profit Factor:          {metrics.get('profit_factor', 0):>10.2f}")
        report.append(f"Payoff Ratio:           {metrics.get('payoff_ratio', 0):>10.2f}")
        report.append(f"Gain Moyen:             {metrics.get('avg_win', 0)*100:>10.2f}%")
        report.append(f"Perte Moyenne:          {metrics.get('avg_loss', 0)*100:>10.2f}%")
        
        # Benchmark
        if benchmark is not None and 'alpha' in metrics:
            report.append("\n🎯 COMPARAISON AU BENCHMARK")
            report.append("-" * 60)
            report.append(f"Alpha:                  {metrics.get('alpha', 0)*100:>10.2f}%")
            report.append(f"Beta:                   {metrics.get('beta', 0):>10.2f}")
            report.append(f"Information Ratio:      {metrics.get('information_ratio', 0):>10.2f}")
            report.append(f"R²:                     {metrics.get('r_squared', 0):>10.2f}")
        
        report.append("\n" + "=" * 60)
        
        full_report = "\n".join(report)
        
        # Sauvegarder si chemin fourni
        if output_path:
            try:
                import os
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(full_report)
                print(f"✅ Rapport sauvegardé: {output_path}")
            except Exception as e:
                print(f"⚠️  Erreur sauvegarde rapport: {e}")
        
        return full_report
    
    @staticmethod
    def compare_strategies(returns_dict: Dict[str, pd.Series],
                          metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Comparer plusieurs stratégies
        
        Args:
            returns_dict: Dict {strategy_name: returns_series}
            metrics_to_compare: Liste des métriques à comparer
            
        Returns:
            DataFrame de comparaison
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                'total_return', 'cagr', 'sharpe', 'sortino', 
                'max_drawdown', 'win_rate', 'profit_factor'
            ]
        
        comparison = {}
        
        for name, returns in returns_dict.items():
            metrics = AdvancedMetrics.calculate_all(returns)
            comparison[name] = {k: metrics.get(k, 0) for k in metrics_to_compare}
        
        df = pd.DataFrame(comparison).T
        df.index.name = 'Strategy'
        
        return df
