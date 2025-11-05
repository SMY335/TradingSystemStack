"""
Complete Portfolio Analysis Runner

Runs comprehensive portfolio risk management and performance attribution analysis.
Combines risk metrics, stress testing, Monte Carlo simulation, and performance attribution.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.portfolio.risk_manager import RiskManager
from src.portfolio.performance_attribution import PerformanceAttributor
from src.portfolio.telegram_alerts import TelegramAlerter


def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load returns data from CSV or generate sample data
    
    Args:
        data_path: Path to CSV file (optional)
        
    Returns:
        DataFrame of returns
    """
    if data_path:
        print(f"Loading data from {data_path}...")
        returns = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(returns)} days of data for {len(returns.columns)} assets")
    else:
        print("Generating sample data...")
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        
        returns_data = {
            'BTC': np.random.normal(0.001, 0.04, len(dates)),
            'ETH': np.random.normal(0.0012, 0.045, len(dates)),
            'SOL': np.random.normal(0.0015, 0.055, len(dates)),
            'AVAX': np.random.normal(0.001, 0.05, len(dates)),
            'MATIC': np.random.normal(0.0008, 0.048, len(dates)),
        }
        returns = pd.DataFrame(returns_data, index=dates)
        print(f"✓ Generated sample data: {len(returns)} days, {len(returns.columns)} assets")
    
    return returns


def run_risk_analysis(returns: pd.DataFrame, weights: dict, output_dir: str = None):
    """
    Run comprehensive risk analysis
    
    Args:
        returns: DataFrame of returns
        weights: Dictionary of portfolio weights
        output_dir: Directory to save reports (optional)
    """
    print("\n" + "=" * 80)
    print("RISK MANAGEMENT ANALYSIS")
    print("=" * 80)
    
    # Initialize risk manager
    risk_manager = RiskManager(returns, weights)
    
    # Calculate risk metrics
    print("\n1. Calculating risk metrics...")
    metrics = risk_manager.calculate_risk_metrics()
    
    print(f"   VaR 95%: {metrics.var_95:.2%}")
    print(f"   CVaR 95%: {metrics.cvar_95:.2%}")
    print(f"   VaR 99%: {metrics.var_99:.2%}")
    print(f"   CVaR 99%: {metrics.cvar_99:.2%}")
    print(f"   Volatility (ann.): {metrics.volatility:.2%}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
    
    # Stress testing
    print("\n2. Running stress tests...")
    stress_results = risk_manager.stress_test()
    for result in stress_results:
        breach = "⚠️ VAR BREACH" if result.var_breach else "✓"
        print(f"   {result.scenario_name}: {result.loss_percentage:.2f}% {breach}")
    
    # Monte Carlo simulation
    print("\n3. Running Monte Carlo simulation (10,000 paths)...")
    final_values, paths = risk_manager.monte_carlo_simulation(
        n_simulations=10000,
        time_horizon=252,
        initial_value=1000000
    )
    print(f"   Expected Value: ${final_values.mean():,.0f}")
    print(f"   VaR 95%: ${np.percentile(final_values, 5):,.0f}")
    print(f"   Best Case (95%): ${np.percentile(final_values, 95):,.0f}")
    print(f"   Worst Case (5%): ${np.percentile(final_values, 5):,.0f}")
    
    # Risk alerts
    print("\n4. Checking risk alerts...")
    alerts = risk_manager.check_risk_alerts()
    
    if any(alerts['critical']):
        print("   🚨 CRITICAL ALERTS:")
        for alert in alerts['critical']:
            print(f"      - {alert}")
    
    if any(alerts['warning']):
        print("   ⚠️  WARNINGS:")
        for alert in alerts['warning']:
            print(f"      - {alert}")
    
    if not any(alerts['critical']) and not any(alerts['warning']):
        print("   ✓ No critical alerts")
    
    # Generate full report
    print("\n5. Generating comprehensive risk report...")
    report = risk_manager.generate_risk_report()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"   ✓ Report saved to {report_file}")
    
    return risk_manager, metrics, alerts


def run_attribution_analysis(
    returns: pd.DataFrame,
    portfolio_weights: dict,
    benchmark_weights: dict = None,
    output_dir: str = None
):
    """
    Run performance attribution analysis
    
    Args:
        returns: DataFrame of returns
        portfolio_weights: Portfolio weights
        benchmark_weights: Benchmark weights (optional)
        output_dir: Directory to save reports
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE ATTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Initialize attributor
    attributor = PerformanceAttributor(returns, portfolio_weights, benchmark_weights)
    
    # Brinson attribution
    print("\n1. Calculating Brinson attribution...")
    attribution = attributor.brinson_attribution()
    
    print(f"   Portfolio Return: {attribution.total_return:.2%}")
    print(f"   Benchmark Return: {attribution.benchmark_return:.2%}")
    print(f"   Active Return: {attribution.active_return:.2%}")
    print(f"   Allocation Effect: {attribution.allocation_effect:.2%}")
    print(f"   Selection Effect: {attribution.selection_effect:.2%}")
    print(f"   Interaction Effect: {attribution.interaction_effect:.2%}")
    
    # Asset contributions
    print("\n2. Asset contributions to return:")
    sorted_contrib = sorted(
        attribution.asset_contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for asset, contrib in sorted_contrib:
        print(f"   {asset}: {contrib:.2%}")
    
    # Factor attribution
    print("\n3. Factor attribution:")
    factor_contrib = attributor.factor_attribution()
    for factor, contrib in factor_contrib.items():
        print(f"   {factor}: {contrib:.4f}")
    
    # Risk attribution
    print("\n4. Risk attribution:")
    risk_attr = attributor.risk_attribution()
    print(f"   Portfolio Risk: {risk_attr.portfolio_risk:.2%}")
    print(f"   Diversification Ratio: {risk_attr.diversification_ratio:.2f}")
    
    print("\n   Risk contributions by asset:")
    sorted_risk = sorted(
        risk_attr.asset_risk_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for asset, risk_contrib in sorted_risk:
        print(f"   {asset}: {risk_contrib:.2%}")
    
    # Risk-adjusted metrics
    print("\n5. Risk-adjusted performance metrics:")
    sharpe = attributor.calculate_sharpe_ratio()
    ir = attributor.calculate_information_ratio()
    sortino = attributor.calculate_sortino_ratio()
    
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Information Ratio: {ir:.2f}")
    print(f"   Sortino Ratio: {sortino:.2f}")
    
    # Generate full report
    print("\n6. Generating attribution report...")
    report = attributor.generate_attribution_report()
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"attribution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"   ✓ Report saved to {report_file}")
    
    return attributor, attribution, risk_attr


def send_telegram_alerts(risk_manager, alerts):
    """
    Send risk alerts via Telegram
    
    Args:
        risk_manager: RiskManager instance
        alerts: Alerts dictionary
    """
    print("\n" + "=" * 80)
    print("TELEGRAM ALERTS")
    print("=" * 80)
    
    alerter = TelegramAlerter()
    
    if not alerter.enabled:
        print("⚠️  Telegram not configured")
        print("   Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables to enable alerts")
        return
    
    # Send critical alerts
    if alerts['critical']:
        print("\nSending critical alerts...")
        for alert in alerts['critical']:
            alerter.send_alert_sync("Critical Risk Alert", alert, 'CRITICAL')
            print(f"   ✓ Sent: {alert}")
    
    # Send summary
    print("\nSending daily summary...")
    metrics = risk_manager.calculate_risk_metrics()
    metrics_dict = {
        'VaR 95%': metrics.var_95,
        'CVaR 95%': metrics.cvar_95,
        'Volatility': metrics.volatility,
        'Max Drawdown': metrics.max_drawdown,
        'Sharpe Ratio': metrics.sharpe_ratio
    }
    
    import asyncio
    asyncio.run(alerter.send_daily_summary(metrics_dict, alerts))
    print("   ✓ Daily summary sent")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run complete portfolio analysis')
    parser.add_argument('--data', type=str, help='Path to returns CSV file')
    parser.add_argument('--output', type=str, default='reports', help='Output directory for reports')
    parser.add_argument('--telegram', action='store_true', help='Send Telegram alerts')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("PORTFOLIO RISK MANAGEMENT & PERFORMANCE ATTRIBUTION")
    print("Complete Analysis Runner")
    print("=" * 80)
    
    # Load data
    returns = load_data(args.data)
    
    # Define portfolio weights (can be customized)
    assets = returns.columns.tolist()
    portfolio_weights = {asset: 1.0 / len(assets) for asset in assets}
    
    # Customize weights for demonstration (optional)
    if len(assets) == 5:
        portfolio_weights = {
            assets[0]: 0.30,  # e.g., BTC
            assets[1]: 0.25,  # e.g., ETH
            assets[2]: 0.20,  # e.g., SOL
            assets[3]: 0.15,  # e.g., AVAX
            assets[4]: 0.10,  # e.g., MATIC
        }
    
    print(f"\nPortfolio Composition:")
    for asset, weight in portfolio_weights.items():
        print(f"  {asset}: {weight:.1%}")
    
    # Run risk analysis
    risk_manager, metrics, alerts = run_risk_analysis(
        returns, portfolio_weights, args.output
    )
    
    # Run attribution analysis
    attributor, attribution, risk_attr = run_attribution_analysis(
        returns, portfolio_weights, output_dir=args.output
    )
    
    # Send Telegram alerts if requested
    if args.telegram:
        send_telegram_alerts(risk_manager, alerts)
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n✓ Risk metrics calculated")
    print(f"✓ Stress tests completed")
    print(f"✓ Monte Carlo simulation run")
    print(f"✓ Performance attribution calculated")
    print(f"✓ Reports generated in {args.output}/")
    
    if alerts['critical']:
        print(f"\n⚠️  {len(alerts['critical'])} critical alerts detected")
    
    print("\nTo view dashboards:")
    print("  Risk Dashboard: streamlit run src/dashboard/risk_dashboard.py")
    print("  Attribution Dashboard: streamlit run src/dashboard/attribution_dashboard.py")
    print()


if __name__ == "__main__":
    main()
