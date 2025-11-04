"""
Simulation Monte Carlo pour analyse de risque
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class MonteCarloSimulator:
    """
    Simulation Monte Carlo pour analyse de risque des stratégies de trading
    
    Utilise bootstrap sampling sur les trades historiques pour:
    - Estimer la distribution des rendements futurs
    - Calculer la probabilité de profit/perte
    - Évaluer le risque de drawdown extrême
    - Identifier les scénarios worst-case
    """
    
    def __init__(self, trades: List[float], n_simulations: int = 10000):
        """
        Args:
            trades: Liste des P&L de chaque trade
            n_simulations: Nombre de simulations Monte Carlo
        """
        self.trades = np.array(trades)
        self.n_simulations = n_simulations
        self.simulations = None
        
        if len(self.trades) == 0:
            raise ValueError("La liste des trades ne peut pas être vide")
    
    def simulate(self, n_trades: Optional[int] = None) -> np.ndarray:
        """
        Simuler des séquences de trades aléatoires via bootstrap
        
        Args:
            n_trades: Nombre de trades à simuler (default: même que historique)
            
        Returns:
            Array (n_simulations, n_trades) des rendements cumulés
        """
        if n_trades is None:
            n_trades = len(self.trades)
        
        # Bootstrap sampling avec remplacement
        simulations = np.zeros((self.n_simulations, n_trades))
        
        for i in range(self.n_simulations):
            # Échantillonner avec remplacement
            sampled_trades = np.random.choice(self.trades, size=n_trades, replace=True)
            # Calculer rendements cumulés
            simulations[i] = np.cumsum(sampled_trades)
        
        self.simulations = simulations
        return simulations
    
    def calculate_risk_metrics(self, simulations: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculer les métriques de risque Monte Carlo
        
        Args:
            simulations: Array des simulations (utilise self.simulations si None)
            
        Returns:
            Dict avec métriques de risque
        """
        if simulations is None:
            if self.simulations is None:
                self.simulate()
            simulations = self.simulations
        
        # Rendements finaux de chaque simulation
        final_returns = simulations[:, -1]
        
        # Drawdowns maximums de chaque simulation
        max_drawdowns = []
        for sim in simulations:
            running_max = np.maximum.accumulate(sim)
            drawdown = (sim - running_max)
            max_drawdowns.append(drawdown.min())
        max_drawdowns = np.array(max_drawdowns)
        
        metrics = {
            # Statistiques de rendement
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'min_return': np.min(final_returns),
            'max_return': np.max(final_returns),
            
            # Percentiles de rendement
            'worst_case_1pct': np.percentile(final_returns, 1),
            'worst_case_5pct': np.percentile(final_returns, 5),
            'worst_case_10pct': np.percentile(final_returns, 10),
            'best_case_90pct': np.percentile(final_returns, 90),
            'best_case_95pct': np.percentile(final_returns, 95),
            'best_case_99pct': np.percentile(final_returns, 99),
            
            # Probabilités
            'prob_profit': np.mean(final_returns > 0) * 100,
            'prob_loss': np.mean(final_returns < 0) * 100,
            'prob_breakeven': np.mean(final_returns == 0) * 100,
            
            # Risque de drawdown
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown_5pct': np.percentile(max_drawdowns, 5),
            'prob_drawdown_10pct': np.mean(max_drawdowns < -0.10) * 100,
            'prob_drawdown_20pct': np.mean(max_drawdowns < -0.20) * 100,
            'prob_drawdown_30pct': np.mean(max_drawdowns < -0.30) * 100,
            
            # Ratio risque/rendement
            'return_risk_ratio': np.mean(final_returns) / np.std(final_returns) if np.std(final_returns) > 0 else 0,
        }
        
        return metrics
    
    def plot_simulations(self, 
                        simulations: Optional[np.ndarray] = None,
                        save_path: str = "reports/monte_carlo.png",
                        n_paths_to_plot: int = 100,
                        figsize: tuple = (14, 8)):
        """
        Visualiser les simulations Monte Carlo
        
        Args:
            simulations: Array des simulations
            save_path: Chemin de sauvegarde
            n_paths_to_plot: Nombre de chemins à afficher
            figsize: Taille de la figure
        """
        if simulations is None:
            if self.simulations is None:
                self.simulate()
            simulations = self.simulations
        
        # Créer dossier si nécessaire
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Simulation Monte Carlo ({self.n_simulations:,} runs)', fontsize=16, fontweight='bold')
        
        # Plot 1: Trajectoires des simulations
        ax1 = axes[0, 0]
        n_to_plot = min(n_paths_to_plot, self.n_simulations)
        for i in range(n_to_plot):
            ax1.plot(simulations[i], alpha=0.1, color='blue', linewidth=0.5)
        
        # Percentiles
        ax1.plot(np.percentile(simulations, 50, axis=0), color='red', linewidth=2, label='Median (P50)')
        ax1.plot(np.percentile(simulations, 5, axis=0), color='orange', linewidth=2, label='P5 (Worst 5%)')
        ax1.plot(np.percentile(simulations, 95, axis=0), color='green', linewidth=2, label='P95 (Best 5%)')
        ax1.fill_between(range(simulations.shape[1]), 
                         np.percentile(simulations, 5, axis=0),
                         np.percentile(simulations, 95, axis=0),
                         alpha=0.2, color='gray', label='P5-P95 Range')
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Monte Carlo Paths')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution des rendements finaux
        ax2 = axes[0, 1]
        final_returns = simulations[:, -1]
        ax2.hist(final_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(final_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_returns):.2f}')
        ax2.axvline(np.median(final_returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_returns):.2f}')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Final Return')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Final Returns')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution des drawdowns maximums
        ax3 = axes[1, 0]
        max_drawdowns = []
        for sim in simulations:
            running_max = np.maximum.accumulate(sim)
            drawdown = sim - running_max
            max_drawdowns.append(drawdown.min())
        
        ax3.hist(max_drawdowns, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax3.axvline(np.mean(max_drawdowns), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean DD: {np.mean(max_drawdowns):.2f}')
        ax3.axvline(np.percentile(max_drawdowns, 5), color='darkred', linestyle='--', linewidth=2,
                   label=f'5th %ile: {np.percentile(max_drawdowns, 5):.2f}')
        ax3.set_xlabel('Maximum Drawdown')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Maximum Drawdowns')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Probabilités cumulées
        ax4 = axes[1, 1]
        sorted_returns = np.sort(final_returns)
        cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
        ax4.plot(sorted_returns, cumulative_prob, linewidth=2, color='blue')
        ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Final Return')
        ax4.set_ylabel('Cumulative Probability (%)')
        ax4.set_title('Cumulative Distribution Function')
        ax4.grid(True, alpha=0.3)
        
        # Annotations
        prob_profit = np.mean(final_returns > 0) * 100
        ax4.text(0.05, 0.95, f'Prob(Profit): {prob_profit:.1f}%', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique Monte Carlo sauvegardé: {save_path}")
    
    def generate_risk_report(self, save_path: Optional[str] = "reports/monte_carlo_report.txt") -> str:
        """
        Générer un rapport texte des métriques de risque
        
        Returns:
            String contenant le rapport
        """
        if self.simulations is None:
            self.simulate()
        
        metrics = self.calculate_risk_metrics()
        
        report = []
        report.append("=" * 70)
        report.append("🎲 RAPPORT MONTE CARLO - ANALYSE DE RISQUE")
        report.append("=" * 70)
        report.append(f"Nombre de simulations: {self.n_simulations:,}")
        report.append(f"Nombre de trades par simulation: {len(self.trades)}")
        report.append(f"Trades historiques utilisés: {len(self.trades)}")
        
        report.append("\n📊 STATISTIQUES DES RENDEMENTS FINAUX")
        report.append("-" * 70)
        report.append(f"Moyenne:                {metrics['mean_return']:>12.2f}")
        report.append(f"Médiane:                {metrics['median_return']:>12.2f}")
        report.append(f"Écart-type:             {metrics['std_return']:>12.2f}")
        report.append(f"Minimum:                {metrics['min_return']:>12.2f}")
        report.append(f"Maximum:                {metrics['max_return']:>12.2f}")
        
        report.append("\n📉 SCÉNARIOS DE RISQUE")
        report.append("-" * 70)
        report.append(f"Pire cas (1%):          {metrics['worst_case_1pct']:>12.2f}")
        report.append(f"Pire cas (5%):          {metrics['worst_case_5pct']:>12.2f}")
        report.append(f"Pire cas (10%):         {metrics['worst_case_10pct']:>12.2f}")
        
        report.append("\n📈 SCÉNARIOS FAVORABLES")
        report.append("-" * 70)
        report.append(f"Meilleur cas (90%):     {metrics['best_case_90pct']:>12.2f}")
        report.append(f"Meilleur cas (95%):     {metrics['best_case_95pct']:>12.2f}")
        report.append(f"Meilleur cas (99%):     {metrics['best_case_99pct']:>12.2f}")
        
        report.append("\n🎯 PROBABILITÉS")
        report.append("-" * 70)
        report.append(f"Probabilité de profit:  {metrics['prob_profit']:>11.1f}%")
        report.append(f"Probabilité de perte:   {metrics['prob_loss']:>11.1f}%")
        
        report.append("\n⚠️  RISQUE DE DRAWDOWN")
        report.append("-" * 70)
        report.append(f"Drawdown moyen:         {metrics['mean_max_drawdown']:>12.2f}")
        report.append(f"Pire DD (5%):           {metrics['worst_drawdown_5pct']:>12.2f}")
        report.append(f"Prob DD > 10%:          {metrics['prob_drawdown_10pct']:>11.1f}%")
        report.append(f"Prob DD > 20%:          {metrics['prob_drawdown_20pct']:>11.1f}%")
        report.append(f"Prob DD > 30%:          {metrics['prob_drawdown_30pct']:>11.1f}%")
        
        report.append("\n📐 RATIOS")
        report.append("-" * 70)
        report.append(f"Return/Risk Ratio:      {metrics['return_risk_ratio']:>12.2f}")
        
        report.append("\n" + "=" * 70)
        
        full_report = "\n".join(report)
        
        # Sauvegarder
        if save_path:
            try:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(full_report)
                print(f"✅ Rapport Monte Carlo sauvegardé: {save_path}")
            except Exception as e:
                print(f"⚠️  Erreur sauvegarde rapport: {e}")
        
        return full_report
    
    def stress_test(self, worst_case_percentile: float = 5) -> Dict[str, Any]:
        """
        Effectuer un stress test basé sur les pires scénarios
        
        Args:
            worst_case_percentile: Percentile pour définir le pire cas (default: 5)
            
        Returns:
            Dict avec résultats du stress test
        """
        if self.simulations is None:
            self.simulate()
        
        final_returns = self.simulations[:, -1]
        worst_case_threshold = np.percentile(final_returns, worst_case_percentile)
        
        # Simulations dans le pire cas
        worst_case_sims = self.simulations[final_returns <= worst_case_threshold]
        
        stress_metrics = {
            'worst_case_threshold': worst_case_threshold,
            'n_worst_case_scenarios': len(worst_case_sims),
            'avg_loss_in_worst_case': np.mean(worst_case_sims[:, -1]),
            'max_loss_in_worst_case': np.min(worst_case_sims[:, -1]),
            'avg_max_dd_in_worst_case': np.mean([
                (sim - np.maximum.accumulate(sim)).min() 
                for sim in worst_case_sims
            ]),
        }
        
        return stress_metrics
