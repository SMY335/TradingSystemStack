"""
Advanced Risk Manager Module

Implements comprehensive risk analysis including:
- VaR (Value at Risk): Historical, Parametric, Monte Carlo
- CVaR (Conditional VaR / Expected Shortfall)
- Stress Testing & Scenario Analysis
- Monte Carlo Simulations
- Correlation Analysis
- Tail Risk Metrics
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.stats import norm, t
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float
    worst_loss: float
    best_gain: float


@dataclass
class StressTestResult:
    """Container for stress test results"""
    scenario_name: str
    portfolio_loss: float
    loss_percentage: float
    var_breach: bool
    asset_impacts: Dict[str, float]


class RiskManager:
    """
    Advanced Risk Manager for portfolio analysis
    
    Features:
    - Multiple VaR calculation methods
    - CVaR (Expected Shortfall)
    - Stress testing with predefined scenarios
    - Monte Carlo simulations
    - Correlation breakdown analysis
    - Tail risk analysis
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        portfolio_weights: Optional[Dict[str, float]] = None,
        confidence_levels: List[float] = [0.95, 0.99],
        risk_free_rate: float = 0.02
    ):
        """
        Initialize Risk Manager
        
        Args:
            returns: DataFrame of asset returns (columns = assets, index = dates)
            portfolio_weights: Dictionary of asset weights (if None, equal weights)
            confidence_levels: List of confidence levels for VaR/CVaR
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        # Validate returns DataFrame
        if returns is None or not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        
        if returns.empty:
            raise ValueError("returns DataFrame cannot be empty")
        
        if len(returns) < 30:
            raise ValueError(f"Insufficient data: {len(returns)} rows. Need at least 30 observations for risk analysis.")
        
        # Check for NaN values
        if returns.isna().any().any():
            logger.warning("returns DataFrame contains NaN values. These will be dropped for calculations.")
        
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        
        if self.n_assets == 0:
            raise ValueError("returns DataFrame must have at least one column (asset)")
        
        # Validate portfolio_weights
        if portfolio_weights is not None:
            if not isinstance(portfolio_weights, dict):
                raise TypeError(f"portfolio_weights must be dict, got {type(portfolio_weights)}")
            
            if not portfolio_weights:
                raise ValueError("portfolio_weights dictionary cannot be empty")
            
            # Check that all weights are numeric and non-negative
            for asset, weight in portfolio_weights.items():
                if not isinstance(weight, (int, float)):
                    raise TypeError(f"Weight for {asset} must be numeric, got {type(weight)}")
                
                if weight < 0:
                    raise ValueError(f"Weight for {asset} cannot be negative, got {weight}")
            
            # Check that all assets in weights exist in returns
            missing_assets = set(portfolio_weights.keys()) - set(self.assets)
            if missing_assets:
                raise ValueError(f"portfolio_weights contains assets not in returns: {missing_assets}")
            
            # Check weight sum is reasonable (allow some tolerance)
            weight_sum = sum(portfolio_weights.values())
            if weight_sum <= 0:
                raise ValueError(f"Sum of portfolio weights must be positive, got {weight_sum}")
            
            if weight_sum < 0.99 or weight_sum > 1.01:
                logger.warning(f"Portfolio weights sum to {weight_sum:.4f}, normalizing to 1.0")
            
            self.weights = portfolio_weights
        else:
            # Set equal weights
            self.weights = {asset: 1.0 / self.n_assets for asset in self.assets}
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Validate confidence_levels
        if not isinstance(confidence_levels, list):
            raise TypeError(f"confidence_levels must be list, got {type(confidence_levels)}")
        
        if not confidence_levels:
            raise ValueError("confidence_levels list cannot be empty")
        
        for level in confidence_levels:
            if not isinstance(level, (int, float)):
                raise TypeError(f"Confidence level must be numeric, got {type(level)}")
            
            if level <= 0 or level >= 1:
                raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        
        self.confidence_levels = confidence_levels
        
        # Validate risk_free_rate
        if not isinstance(risk_free_rate, (int, float)):
            raise TypeError(f"risk_free_rate must be numeric, got {type(risk_free_rate)}")
        
        if risk_free_rate < -0.1 or risk_free_rate > 0.5:
            raise ValueError(f"risk_free_rate seems unrealistic: {risk_free_rate}. Expected range: -0.1 to 0.5")
        
        self.risk_free_rate = risk_free_rate
        
        # Calculate portfolio returns
        self.portfolio_returns = self._calculate_portfolio_returns()
        
        # Calculate correlation matrix
        self.correlation_matrix = returns.corr()
        self.covariance_matrix = returns.cov()
        
        logger.info(f"RiskManager initialized: {self.n_assets} assets, {len(returns)} observations")
        
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns based on weights"""
        weights_array = np.array([self.weights[asset] for asset in self.assets])
        return (self.returns * weights_array).sum(axis=1)
    
    def calculate_var_historical(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Historical VaR
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value (positive number represents potential loss)
        """
        # Validate confidence_level
        if not isinstance(confidence_level, (int, float)):
            raise TypeError(f"confidence_level must be numeric, got {type(confidence_level)}")
        
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
        
        alpha = 1 - confidence_level
        var = -np.percentile(self.portfolio_returns, alpha * 100)
        return var
    
    def calculate_var_parametric(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Parametric VaR (assuming normal distribution)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value
        """
        # Validate confidence_level
        if not isinstance(confidence_level, (int, float)):
            raise TypeError(f"confidence_level must be numeric, got {type(confidence_level)}")
        
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
        
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        z_score = norm.ppf(1 - confidence_level)
        var = -(mu + sigma * z_score)
        return var
    
    def calculate_var_monte_carlo(
        self,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
        time_horizon: int = 1
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            confidence_level: Confidence level
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Tuple of (VaR value, simulated returns array)
        """
        # Validate confidence_level
        if not isinstance(confidence_level, (int, float)):
            raise TypeError(f"confidence_level must be numeric, got {type(confidence_level)}")
        
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
        
        # Validate n_simulations
        if not isinstance(n_simulations, int):
            raise TypeError(f"n_simulations must be int, got {type(n_simulations)}")
        
        if n_simulations < 100:
            raise ValueError(f"n_simulations too low: {n_simulations}. Need at least 100 for meaningful results.")
        
        if n_simulations > 1000000:
            raise ValueError(f"n_simulations too high: {n_simulations}. Maximum allowed: 1,000,000")
        
        # Validate time_horizon
        if not isinstance(time_horizon, int):
            raise TypeError(f"time_horizon must be int, got {type(time_horizon)}")
        
        if time_horizon < 1:
            raise ValueError(f"time_horizon must be at least 1 day, got {time_horizon}")
        
        if time_horizon > 365:
            raise ValueError(f"time_horizon too long: {time_horizon} days. Maximum: 365 days")
        
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        
        # Simulate returns
        simulated_returns = np.random.normal(
            mu * time_horizon,
            sigma * np.sqrt(time_horizon),
            n_simulations
        )
        
        # Calculate VaR
        alpha = 1 - confidence_level
        var = -np.percentile(simulated_returns, alpha * 100)
        
        return var, simulated_returns
    
    def calculate_cvar(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        Args:
            confidence_level: Confidence level
            method: 'historical' or 'parametric'
            
        Returns:
            CVaR value
        """
        # Validate confidence_level
        if not isinstance(confidence_level, (int, float)):
            raise TypeError(f"confidence_level must be numeric, got {type(confidence_level)}")
        
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
        
        # Validate method
        valid_methods = ['historical', 'parametric']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
        
        if method == 'historical':
            var = self.calculate_var_historical(confidence_level)
            # Average of losses beyond VaR
            losses_beyond_var = self.portfolio_returns[self.portfolio_returns <= -var]
            cvar = -losses_beyond_var.mean() if len(losses_beyond_var) > 0 else var
            
        elif method == 'parametric':
            mu = self.portfolio_returns.mean()
            sigma = self.portfolio_returns.std()
            alpha = 1 - confidence_level
            z_alpha = norm.ppf(alpha)
            cvar = -(mu - sigma * norm.pdf(z_alpha) / alpha)
            
        return cvar
    
    def stress_test(
        self,
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[StressTestResult]:
        """
        Perform stress testing with various scenarios
        
        Args:
            scenarios: Dictionary of scenarios with asset-level shocks
                      Format: {'scenario_name': {'asset': shock_percentage}}
                      If None, uses predefined scenarios
                      
        Returns:
            List of StressTestResult objects
        """
        if scenarios is None:
            scenarios = self._get_default_scenarios()
        
        results = []
        var_95 = self.calculate_var_historical(0.95)
        
        for scenario_name, shocks in scenarios.items():
            # Calculate portfolio impact
            portfolio_loss = 0.0
            asset_impacts = {}
            
            for asset in self.assets:
                shock = shocks.get(asset, 0.0)
                impact = self.weights[asset] * shock
                portfolio_loss += impact
                asset_impacts[asset] = impact
            
            # Check if VaR is breached
            var_breach = abs(portfolio_loss) > var_95
            
            result = StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss=portfolio_loss,
                loss_percentage=portfolio_loss * 100,
                var_breach=var_breach,
                asset_impacts=asset_impacts
            )
            results.append(result)
        
        return results
    
    def _get_default_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Get predefined stress test scenarios"""
        scenarios = {
            'Market Crash -20%': {asset: -0.20 for asset in self.assets},
            'Market Crash -50%': {asset: -0.50 for asset in self.assets},
            'Extreme Volatility Spike': {
                asset: -0.15 if i % 2 == 0 else 0.05
                for i, asset in enumerate(self.assets)
            },
            'Correlation Breakdown': {
                asset: (-1) ** i * 0.25
                for i, asset in enumerate(self.assets)
            },
            'Flash Crash': {asset: -0.30 for asset in self.assets},
            'Crypto Winter': {asset: -0.65 for asset in self.assets},
        }
        return scenarios
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = 10000,
        time_horizon: int = 252,
        initial_value: float = 1000000.0
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Run Monte Carlo simulation for portfolio evolution
        
        Args:
            n_simulations: Number of simulation paths
            time_horizon: Number of days to simulate
            initial_value: Initial portfolio value
            
        Returns:
            Tuple of (final values array, simulation paths DataFrame)
        """
        # Validate n_simulations
        if not isinstance(n_simulations, int):
            raise TypeError(f"n_simulations must be int, got {type(n_simulations)}")
        
        if n_simulations < 100:
            raise ValueError(f"n_simulations too low: {n_simulations}. Need at least 100.")
        
        if n_simulations > 1000000:
            raise ValueError(f"n_simulations too high: {n_simulations}. Maximum: 1,000,000")
        
        # Validate time_horizon
        if not isinstance(time_horizon, int):
            raise TypeError(f"time_horizon must be int, got {type(time_horizon)}")
        
        if time_horizon < 1:
            raise ValueError(f"time_horizon must be at least 1 day, got {time_horizon}")
        
        if time_horizon > 365 * 5:
            raise ValueError(f"time_horizon too long: {time_horizon} days. Maximum: 1,825 days (5 years)")
        
        # Validate initial_value
        if not isinstance(initial_value, (int, float)):
            raise TypeError(f"initial_value must be numeric, got {type(initial_value)}")
        
        if initial_value <= 0:
            raise ValueError(f"initial_value must be positive, got {initial_value}")
        
        mu = self.portfolio_returns.mean()
        sigma = self.portfolio_returns.std()
        
        # Generate daily returns
        dt = 1  # Daily
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Simulate paths
        daily_returns = np.random.normal(drift, diffusion, (time_horizon, n_simulations))
        
        # Calculate cumulative returns
        price_paths = initial_value * np.exp(np.cumsum(daily_returns, axis=0))
        
        # Add initial value
        price_paths = np.vstack([np.full(n_simulations, initial_value), price_paths])
        
        # Create DataFrame for selected paths (for visualization)
        n_display = min(100, n_simulations)  # Display max 100 paths
        paths_df = pd.DataFrame(
            price_paths[:, :n_display],
            columns=[f'Path_{i}' for i in range(n_display)]
        )
        
        final_values = price_paths[-1, :]
        
        return final_values, paths_df
    
    def calculate_tail_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate tail risk metrics
        
        Returns:
            Dictionary with tail risk metrics
        """
        returns = self.portfolio_returns.dropna()
        
        metrics = {
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'excess_kurtosis': stats.kurtosis(returns, fisher=True),
            'jarque_bera_stat': stats.jarque_bera(returns)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns)[1],
            'worst_1pct': np.percentile(returns, 1),
            'worst_5pct': np.percentile(returns, 5),
            'best_99pct': np.percentile(returns, 99),
            'best_95pct': np.percentile(returns, 95),
        }
        
        return metrics
    
    def analyze_correlation_breakdown(
        self,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Analyze correlation breakdown over time
        
        Args:
            window: Rolling window size in days
            
        Returns:
            DataFrame with rolling correlations
        """
        # Validate window
        if not isinstance(window, int):
            raise TypeError(f"window must be int, got {type(window)}")
        
        if window < 10:
            raise ValueError(f"window too small: {window}. Need at least 10 observations.")
        
        if window > len(self.returns):
            raise ValueError(f"window ({window}) larger than available data ({len(self.returns)})")
        
        rolling_corr = self.returns.rolling(window=window).corr()
        return rolling_corr
    
    def calculate_max_drawdown(self) -> Tuple[float, pd.Series]:
        """
        Calculate maximum drawdown
        
        Returns:
            Tuple of (max drawdown, drawdown series)
        """
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return max_drawdown, drawdown
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Returns:
            RiskMetrics object with all metrics
        """
        returns = self.portfolio_returns.dropna()
        
        # VaR and CVaR
        var_95 = self.calculate_var_historical(0.95)
        var_99 = self.calculate_var_historical(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        cvar_99 = self.calculate_cvar(0.99)
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Drawdown
        max_dd, _ = self.calculate_max_drawdown()
        
        # Sharpe Ratio
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        sharpe = excess_returns / volatility if volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = excess_returns / downside_std if downside_std > 0 else 0
        
        # Tail metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Worst/Best
        worst_loss = returns.min()
        best_gain = returns.max()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            skewness=skewness,
            kurtosis=kurtosis,
            worst_loss=worst_loss,
            best_gain=best_gain
        )
    
    def check_risk_alerts(
        self,
        var_threshold: float = 0.05,
        drawdown_threshold: float = -0.20,
        correlation_threshold: float = 0.95
    ) -> Dict[str, List[str]]:
        """
        Check for risk alerts
        
        Args:
            var_threshold: VaR threshold (as decimal)
            drawdown_threshold: Maximum acceptable drawdown
            correlation_threshold: Correlation threshold for alert
            
        Returns:
            Dictionary with alert types and messages
        """
        # Validate var_threshold
        if not isinstance(var_threshold, (int, float)):
            raise TypeError(f"var_threshold must be numeric, got {type(var_threshold)}")
        
        if var_threshold <= 0 or var_threshold > 1:
            raise ValueError(f"var_threshold must be between 0 and 1, got {var_threshold}")
        
        # Validate drawdown_threshold
        if not isinstance(drawdown_threshold, (int, float)):
            raise TypeError(f"drawdown_threshold must be numeric, got {type(drawdown_threshold)}")
        
        if drawdown_threshold >= 0 or drawdown_threshold < -1:
            raise ValueError(f"drawdown_threshold must be between -1 and 0, got {drawdown_threshold}")
        
        # Validate correlation_threshold
        if not isinstance(correlation_threshold, (int, float)):
            raise TypeError(f"correlation_threshold must be numeric, got {type(correlation_threshold)}")
        
        if correlation_threshold < 0 or correlation_threshold > 1:
            raise ValueError(f"correlation_threshold must be between 0 and 1, got {correlation_threshold}")
        
        alerts = {
            'critical': [],
            'warning': [],
            'info': []
        }
        
        # VaR check
        var_95 = self.calculate_var_historical(0.95)
        if var_95 > var_threshold:
            alerts['critical'].append(
                f"VaR 95% ({var_95:.2%}) exceeds threshold ({var_threshold:.2%})"
            )
        
        # Drawdown check
        max_dd, _ = self.calculate_max_drawdown()
        if max_dd < drawdown_threshold:
            alerts['critical'].append(
                f"Max drawdown ({max_dd:.2%}) exceeds threshold ({drawdown_threshold:.2%})"
            )
        
        # Correlation check
        corr_matrix = self.correlation_matrix
        high_corr_pairs = []
        for i in range(len(self.assets)):
            for j in range(i + 1, len(self.assets)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > correlation_threshold:
                    high_corr_pairs.append(
                        (self.assets[i], self.assets[j], corr)
                    )
        
        if high_corr_pairs:
            alerts['warning'].append(
                f"High correlation detected: {len(high_corr_pairs)} pairs > {correlation_threshold}"
            )
            for asset1, asset2, corr in high_corr_pairs[:3]:  # Show top 3
                alerts['info'].append(f"{asset1}-{asset2}: {corr:.3f}")
        
        # Tail risk check
        tail_metrics = self.calculate_tail_risk_metrics()
        if abs(tail_metrics['skewness']) > 1:
            alerts['warning'].append(
                f"High skewness detected: {tail_metrics['skewness']:.3f}"
            )
        
        if tail_metrics['excess_kurtosis'] > 3:
            alerts['warning'].append(
                f"High kurtosis detected: {tail_metrics['excess_kurtosis']:.3f} (fat tails)"
            )
        
        return alerts
    
    def generate_risk_report(self) -> str:
        """
        Generate comprehensive risk report
        
        Returns:
            Formatted risk report string
        """
        metrics = self.calculate_risk_metrics()
        alerts = self.check_risk_alerts()
        stress_results = self.stress_test()
        
        report = ["=" * 80]
        report.append("PORTFOLIO RISK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Portfolio composition
        report.append("Portfolio Composition:")
        for asset, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {asset}: {weight:.2%}")
        report.append("")
        
        # Risk metrics
        report.append("Risk Metrics:")
        report.append(f"  VaR 95%: {metrics.var_95:.4f} ({metrics.var_95*100:.2f}%)")
        report.append(f"  VaR 99%: {metrics.var_99:.4f} ({metrics.var_99*100:.2f}%)")
        report.append(f"  CVaR 95%: {metrics.cvar_95:.4f} ({metrics.cvar_95*100:.2f}%)")
        report.append(f"  CVaR 99%: {metrics.cvar_99:.4f} ({metrics.cvar_99*100:.2f}%)")
        report.append(f"  Volatility (ann.): {metrics.volatility:.4f} ({metrics.volatility*100:.2f}%)")
        report.append(f"  Max Drawdown: {metrics.max_drawdown:.4f} ({metrics.max_drawdown*100:.2f}%)")
        report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        report.append(f"  Sortino Ratio: {metrics.sortino_ratio:.4f}")
        report.append(f"  Skewness: {metrics.skewness:.4f}")
        report.append(f"  Kurtosis: {metrics.kurtosis:.4f}")
        report.append("")
        
        # Stress test results
        report.append("Stress Test Results:")
        for result in stress_results:
            breach_flag = " ⚠️ VAR BREACH" if result.var_breach else ""
            report.append(f"  {result.scenario_name}: {result.loss_percentage:.2f}%{breach_flag}")
        report.append("")
        
        # Alerts
        if any(alerts.values()):
            report.append("Risk Alerts:")
            for level, messages in alerts.items():
                if messages:
                    report.append(f"  [{level.upper()}]")
                    for msg in messages:
                        report.append(f"    - {msg}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def example_usage():
    """Example usage of RiskManager"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    n_assets = 5
    
    # Simulate returns with different characteristics
    returns_data = {}
    for i in range(n_assets):
        mu = 0.0005 * (i + 1)
        sigma = 0.02 * (i + 1)
        returns_data[f'Asset_{i+1}'] = np.random.normal(mu, sigma, len(dates))
    
    returns = pd.DataFrame(returns_data, index=dates)
    
    # Initialize risk manager
    weights = {f'Asset_{i+1}': 1.0 / n_assets for i in range(n_assets)}
    risk_manager = RiskManager(returns, weights)
    
    # Generate report
    print(risk_manager.generate_risk_report())
    
    # Monte Carlo simulation
    final_values, paths = risk_manager.monte_carlo_simulation(n_simulations=10000)
    print(f"\nMonte Carlo Results (10,000 simulations):")
    print(f"  Expected Final Value: ${final_values.mean():,.2f}")
    print(f"  Std Dev: ${final_values.std():,.2f}")
    print(f"  95% VaR: ${np.percentile(final_values, 5):,.2f}")


if __name__ == "__main__":
    example_usage()
