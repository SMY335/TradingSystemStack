# Risk Management Guide

Complete guide for using the advanced risk management system with Riskfolio-Lib integration.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Risk Metrics](#risk-metrics)
- [VaR & CVaR Calculations](#var--cvar-calculations)
- [Stress Testing](#stress-testing)
- [Monte Carlo Simulation](#monte-carlo-simulation)
- [Risk Alerts](#risk-alerts)
- [Interactive Dashboard](#interactive-dashboard)
- [API Reference](#api-reference)

## Overview

The Risk Management module provides comprehensive portfolio risk analysis capabilities including:
- Multiple VaR calculation methods (Historical, Parametric, Monte Carlo)
- CVaR (Conditional VaR / Expected Shortfall)
- Stress testing with predefined and custom scenarios
- Monte Carlo simulations (10,000+ paths)
- Correlation breakdown analysis
- Tail risk metrics
- Automated risk alerts via Telegram

## Features

### 1. Value at Risk (VaR)
Calculate portfolio VaR using three different methods:
- **Historical VaR**: Based on actual historical returns
- **Parametric VaR**: Assumes normal distribution
- **Monte Carlo VaR**: Simulation-based approach

### 2. Conditional VaR (CVaR)
Also known as Expected Shortfall, CVaR measures the average loss in the worst-case scenarios beyond the VaR threshold.

### 3. Stress Testing
Test portfolio resilience under extreme market conditions:
- Market crash scenarios (-20%, -50%)
- Volatility spikes
- Correlation breakdowns
- Flash crashes
- Crypto winter scenarios
- Custom scenarios

### 4. Monte Carlo Simulation
Simulate thousands of potential portfolio paths to understand:
- Distribution of potential outcomes
- Probability of reaching targets
- Worst-case scenarios
- Risk of ruin

### 5. Tail Risk Analysis
Analyze extreme events and fat-tail behavior:
- Skewness
- Kurtosis
- Jarque-Bera test
- Extreme percentiles

## Quick Start

### Basic Usage

```python
from src.portfolio.risk_manager import RiskManager
import pandas as pd

# Load your returns data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Define portfolio weights
weights = {
    'BTC': 0.40,
    'ETH': 0.30,
    'SOL': 0.20,
    'AVAX': 0.10
}

# Initialize Risk Manager
risk_manager = RiskManager(returns, weights)

# Calculate comprehensive risk metrics
metrics = risk_manager.calculate_risk_metrics()
print(f"VaR 95%: {metrics.var_95:.2%}")
print(f"CVaR 95%: {metrics.cvar_95:.2%}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
```

### Generate Full Risk Report

```python
# Generate comprehensive risk report
report = risk_manager.generate_risk_report()
print(report)
```

## Risk Metrics

### Volatility
Annualized portfolio volatility:
```python
volatility = risk_manager.portfolio_returns.std() * np.sqrt(252)
```

### Maximum Drawdown
Largest peak-to-trough decline:
```python
max_dd, drawdown_series = risk_manager.calculate_max_drawdown()
```

### Sharpe Ratio
Risk-adjusted return metric:
```python
sharpe = risk_manager.calculate_sharpe_ratio(risk_free_rate=0.02)
```

### Sortino Ratio
Similar to Sharpe but only considers downside volatility:
```python
sortino = risk_manager.calculate_sortino_ratio(risk_free_rate=0.02)
```

## VaR & CVaR Calculations

### Historical VaR
```python
# 95% confidence level
var_95 = risk_manager.calculate_var_historical(0.95)
print(f"Historical VaR 95%: {var_95:.2%}")

# 99% confidence level
var_99 = risk_manager.calculate_var_historical(0.99)
print(f"Historical VaR 99%: {var_99:.2%}")
```

### Parametric VaR
```python
var_parametric = risk_manager.calculate_var_parametric(0.95)
print(f"Parametric VaR 95%: {var_parametric:.2%}")
```

### Monte Carlo VaR
```python
var_mc, simulations = risk_manager.calculate_var_monte_carlo(
    confidence_level=0.95,
    n_simulations=10000,
    time_horizon=1
)
print(f"Monte Carlo VaR 95%: {var_mc:.2%}")
```

### CVaR (Expected Shortfall)
```python
# Historical method
cvar_95 = risk_manager.calculate_cvar(0.95, method='historical')
print(f"CVaR 95%: {cvar_95:.2%}")

# Parametric method
cvar_parametric = risk_manager.calculate_cvar(0.95, method='parametric')
```

## Stress Testing

### Using Default Scenarios
```python
# Run predefined stress tests
stress_results = risk_manager.stress_test()

for result in stress_results:
    print(f"Scenario: {result.scenario_name}")
    print(f"  Loss: {result.loss_percentage:.2f}%")
    print(f"  VaR Breach: {result.var_breach}")
    print(f"  Asset Impacts: {result.asset_impacts}")
```

### Custom Scenarios
```python
# Define custom stress scenario
custom_scenarios = {
    'Black Swan Event': {
        'BTC': -0.40,  # -40% shock
        'ETH': -0.45,  # -45% shock
        'SOL': -0.60,  # -60% shock
        'AVAX': -0.55  # -55% shock
    },
    'Regulatory Crackdown': {
        'BTC': -0.25,
        'ETH': -0.30,
        'SOL': -0.35,
        'AVAX': -0.40
    }
}

results = risk_manager.stress_test(custom_scenarios)
```

## Monte Carlo Simulation

### Running Simulations
```python
# Simulate 10,000 portfolio paths over 1 year
final_values, paths = risk_manager.monte_carlo_simulation(
    n_simulations=10000,
    time_horizon=252,  # Trading days
    initial_value=1000000  # $1M initial portfolio
)

# Analyze results
import numpy as np
print(f"Expected Value: ${final_values.mean():,.0f}")
print(f"Median: ${np.median(final_values):,.0f}")
print(f"VaR 95%: ${np.percentile(final_values, 5):,.0f}")
print(f"Best Case (95%): ${np.percentile(final_values, 95):,.0f}")
```

### Probability Analysis
```python
# Probability of reaching target
target = 1500000  # $1.5M target
prob_success = (final_values >= target).sum() / len(final_values)
print(f"Probability of reaching ${target:,.0f}: {prob_success:.1%}")

# Probability of loss
prob_loss = (final_values < initial_value).sum() / len(final_values)
print(f"Probability of loss: {prob_loss:.1%}")
```

## Risk Alerts

### Checking Alerts
```python
# Check for risk threshold breaches
alerts = risk_manager.check_risk_alerts(
    var_threshold=0.05,  # 5% VaR threshold
    drawdown_threshold=-0.20,  # -20% drawdown threshold
    correlation_threshold=0.95  # 95% correlation threshold
)

# Display alerts
if alerts['critical']:
    print("🚨 CRITICAL ALERTS:")
    for alert in alerts['critical']:
        print(f"  - {alert}")

if alerts['warning']:
    print("⚠️  WARNINGS:")
    for alert in alerts['warning']:
        print(f"  - {alert}")
```

### Telegram Integration
```python
from src.portfolio.telegram_alerts import TelegramAlerter

# Set up environment variables first:
# export TELEGRAM_BOT_TOKEN='your_bot_token'
# export TELEGRAM_CHAT_ID='your_chat_id'

alerter = TelegramAlerter()

# Test connection
import asyncio
asyncio.run(alerter.test_connection())

# Send VaR breach alert
if metrics.var_95 > 0.05:
    asyncio.run(alerter.send_var_breach_alert(
        var_level=metrics.var_95,
        threshold=0.05,
        confidence=0.95
    ))

# Send daily summary
asyncio.run(alerter.send_daily_summary(
    metrics={
        'VaR 95%': metrics.var_95,
        'CVaR 95%': metrics.cvar_95,
        'Volatility': metrics.volatility,
        'Sharpe Ratio': metrics.sharpe_ratio
    },
    alerts=alerts
))
```

## Interactive Dashboard

### Launching the Dashboard
```bash
# Start the Risk Dashboard
streamlit run src/dashboard/risk_dashboard.py
```

### Dashboard Features

1. **Risk Analysis Tab**
   - Key risk metrics display
   - VaR & CVaR comparison charts
   - Returns distribution analysis
   - Correlation heatmap
   - Historical drawdown chart
   - Tail risk metrics
   - Real-time risk alerts

2. **Monte Carlo Tab**
   - Configurable simulation parameters
   - Distribution of final values
   - Sample simulation paths
   - Probability analysis

3. **Stress Testing Tab**
   - Predefined scenario results
   - Asset-level impact analysis
   - VaR breach indicators
   - Interactive scenario selection

4. **Scenario Analysis Tab**
   - Custom scenario builder
   - Asset-level shock inputs
   - Impact visualization
   - Contribution analysis

## API Reference

### RiskManager Class

#### Initialization
```python
RiskManager(
    returns: pd.DataFrame,
    portfolio_weights: Optional[Dict[str, float]] = None,
    confidence_levels: List[float] = [0.95, 0.99],
    risk_free_rate: float = 0.02
)
```

#### Key Methods

**calculate_risk_metrics() -> RiskMetrics**
Calculate comprehensive risk metrics including VaR, CVaR, volatility, drawdown, and ratios.

**calculate_var_historical(confidence_level: float) -> float**
Calculate historical VaR at specified confidence level.

**calculate_var_parametric(confidence_level: float) -> float**
Calculate parametric VaR assuming normal distribution.

**calculate_var_monte_carlo(confidence_level, n_simulations, time_horizon) -> Tuple**
Calculate VaR using Monte Carlo simulation.

**calculate_cvar(confidence_level, method='historical') -> float**
Calculate Conditional VaR (Expected Shortfall).

**stress_test(scenarios=None) -> List[StressTestResult]**
Run stress tests with predefined or custom scenarios.

**monte_carlo_simulation(n_simulations, time_horizon, initial_value) -> Tuple**
Run Monte Carlo simulation for portfolio evolution.

**check_risk_alerts(var_threshold, drawdown_threshold, correlation_threshold) -> Dict**
Check for risk threshold breaches and generate alerts.

**generate_risk_report() -> str**
Generate comprehensive text risk report.

## Best Practices

1. **Regular Monitoring**: Run risk analysis daily or after significant market moves
2. **Multiple Methods**: Compare VaR across different calculation methods
3. **Stress Testing**: Regularly test portfolio against extreme scenarios
4. **Alert Thresholds**: Set appropriate thresholds based on risk tolerance
5. **Documentation**: Keep records of risk metrics and major events
6. **Backtesting**: Validate VaR models by comparing predictions to actual losses

## Example Workflow

```python
# Complete risk management workflow
from src.portfolio.risk_manager import RiskManager
import pandas as pd

# 1. Load data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
weights = {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1}

# 2. Initialize risk manager
rm = RiskManager(returns, weights)

# 3. Calculate metrics
metrics = rm.calculate_risk_metrics()

# 4. Run stress tests
stress_results = rm.stress_test()

# 5. Monte Carlo simulation
final_values, paths = rm.monte_carlo_simulation(n_simulations=10000)

# 6. Check alerts
alerts = rm.check_risk_alerts()

# 7. Generate report
report = rm.generate_risk_report()
print(report)

# 8. Send alerts if needed
if alerts['critical']:
    # Send Telegram alerts
    pass
```

## Troubleshooting

### Common Issues

**Issue**: VaR seems too high/low
- **Solution**: Check data quality, ensure returns are in decimal format (not %), verify weights sum to 1

**Issue**: Monte Carlo simulation is slow
- **Solution**: Reduce n_simulations or time_horizon, ensure data is properly formatted

**Issue**: Telegram alerts not working
- **Solution**: Verify TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables are set correctly

## References

- [Riskfolio-Lib Documentation](https://riskfolio-lib.readthedocs.io/)
- [VaR and CVaR](https://www.investopedia.com/terms/v/var.asp)
- [Stress Testing Best Practices](https://www.bis.org/publ/bcbs155.pdf)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/TradingSystemStack/issues
- Documentation: docs/portfolio/
