# Performance Attribution Guide

Complete guide for using the advanced performance attribution system.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Brinson Attribution](#brinson-attribution)
- [Factor Attribution](#factor-attribution)
- [Asset Contributions](#asset-contributions)
- [Risk Attribution](#risk-attribution)
- [Time-Weighted vs Money-Weighted Returns](#time-weighted-vs-money-weighted-returns)
- [Rolling Attribution](#rolling-attribution)
- [Interactive Dashboard](#interactive-dashboard)
- [API Reference](#api-reference)

## Overview

The Performance Attribution module provides comprehensive analysis of portfolio performance, helping you understand:
- **Where** returns came from (which assets)
- **Why** portfolio outperformed/underperformed benchmark
- **How** to improve allocation and selection decisions
- **What** risks each asset contributes

## Features

### 1. Brinson Attribution
Decompose active returns into:
- **Allocation Effect**: Impact of over/underweighting assets
- **Selection Effect**: Impact of asset performance vs benchmark
- **Interaction Effect**: Combined allocation and selection effects

### 2. Factor Attribution
Analyze returns through factor lens:
- Market factor
- Momentum factor
- Volatility factor
- Custom factors
- Alpha generation

### 3. Risk Attribution
Understand risk sources:
- Risk contribution by asset
- Marginal risk contributions
- Component VaR
- Diversification ratio

### 4. Time-Series Analysis
- Rolling attribution over time
- Performance trends
- Consistency analysis

## Quick Start

### Basic Usage

```python
from src.portfolio.performance_attribution import PerformanceAttributor
import pandas as pd

# Load returns data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Define portfolio and benchmark weights
portfolio_weights = {
    'BTC': 0.40,
    'ETH': 0.30,
    'SOL': 0.20,
    'AVAX': 0.10
}

benchmark_weights = {
    'BTC': 0.25,
    'ETH': 0.25,
    'SOL': 0.25,
    'AVAX': 0.25
}

# Initialize Performance Attributor
attributor = PerformanceAttributor(
    returns,
    portfolio_weights,
    benchmark_weights
)

# Calculate Brinson attribution
attribution = attributor.brinson_attribution()

print(f"Portfolio Return: {attribution.total_return:.2%}")
print(f"Benchmark Return: {attribution.benchmark_return:.2%}")
print(f"Active Return: {attribution.active_return:.2%}")
print(f"Allocation Effect: {attribution.allocation_effect:.2%}")
print(f"Selection Effect: {attribution.selection_effect:.2%}")
```

### Generate Full Attribution Report

```python
# Generate comprehensive attribution report
report = attributor.generate_attribution_report()
print(report)
```

## Brinson Attribution

Brinson-Fachler attribution decomposes portfolio active return into three components:

### Allocation Effect
Measures the impact of portfolio weight decisions:
```
Allocation = (Weight_Portfolio - Weight_Benchmark) × Return_Benchmark
```

**Interpretation**: Positive allocation effect means you overweighted assets that outperformed the benchmark average.

### Selection Effect
Measures the impact of asset selection within sectors:
```
Selection = Weight_Benchmark × (Return_Asset - Return_Benchmark)
```

**Interpretation**: Positive selection effect means your assets outperformed their benchmark weights.

### Interaction Effect
Captures the combined effect of allocation and selection:
```
Interaction = (Weight_Portfolio - Weight_Benchmark) × (Return_Asset - Return_Benchmark)
```

### Example

```python
# Full period attribution
attribution = attributor.brinson_attribution()

print("\n=== Brinson Attribution ===")
print(f"Total Return: {attribution.total_return:.2%}")
print(f"Benchmark Return: {attribution.benchmark_return:.2%}")
print(f"Active Return: {attribution.active_return:.2%}")
print(f"\nDecomposition:")
print(f"  Allocation: {attribution.allocation_effect:.2%}")
print(f"  Selection: {attribution.selection_effect:.2%}")
print(f"  Interaction: {attribution.interaction_effect:.2%}")

# Asset-level contributions
print(f"\nAsset Contributions:")
for asset, contrib in attribution.asset_contributions.items():
    print(f"  {asset}: {contrib:.2%}")
```

### Period-Specific Attribution

```python
# Attribution for specific time period
attribution_2023 = attributor.brinson_attribution(
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## Factor Attribution

Analyze performance through factor lens using regression analysis.

### Using Synthetic Factors

```python
# Use built-in synthetic factors
factor_contrib = attributor.factor_attribution()

print("\n=== Factor Attribution ===")
for factor, contrib in factor_contrib.items():
    print(f"{factor}: {contrib:.4f}")
```

The system creates three synthetic factors:
1. **Market**: Equal-weighted market average
2. **Momentum**: 12-1 month momentum
3. **Volatility**: Rolling volatility factor

### Custom Factors

```python
# Define custom factors
import pandas as pd

custom_factors = pd.DataFrame({
    'Market': market_returns,
    'Value': value_factor_returns,
    'Size': size_factor_returns,
    'Quality': quality_factor_returns
}, index=returns.index)

# Run attribution with custom factors
factor_contrib = attributor.factor_attribution(custom_factors)
```

### Interpretation

```python
# Analyze factor exposures
for factor, contrib in factor_contrib.items():
    if factor == 'Alpha':
        print(f"Alpha (skill): {contrib:.4f}")
    else:
        print(f"{factor} exposure contribution: {contrib:.4f}")
```

## Asset Contributions

Understand how each asset contributed to portfolio performance.

### Full Period Contributions

```python
# Total contribution over full period
full_contrib = attributor.calculate_asset_contributions(period='full')
print(full_contrib)
```

### Daily Contributions

```python
# Daily asset contributions
daily_contrib = attributor.calculate_asset_contributions(period='daily')

# Plot cumulative contributions
import matplotlib.pyplot as plt
daily_contrib.cumsum().plot(figsize=(12, 6))
plt.title('Cumulative Asset Contributions')
plt.show()
```

### Monthly Contributions

```python
# Monthly asset contributions
monthly_contrib = attributor.calculate_asset_contributions(period='monthly')

# Analyze best/worst months per asset
for asset in monthly_contrib.columns:
    best_month = monthly_contrib[asset].idxmax()
    worst_month = monthly_contrib[asset].idxmin()
    print(f"{asset}:")
    print(f"  Best: {best_month} ({monthly_contrib.loc[best_month, asset]:.2%})")
    print(f"  Worst: {worst_month} ({monthly_contrib.loc[worst_month, asset]:.2%})")
```

### Yearly Contributions

```python
# Yearly contributions
yearly_contrib = attributor.calculate_asset_contributions(period='yearly')
```

## Risk Attribution

Understand how each asset contributes to portfolio risk.

### Complete Risk Analysis

```python
# Calculate risk attribution
risk_attr = attributor.risk_attribution()

print("\n=== Risk Attribution ===")
print(f"Portfolio Risk: {risk_attr.portfolio_risk:.2%}")
print(f"Diversification Ratio: {risk_attr.diversification_ratio:.2f}")

print("\nRisk Contributions:")
for asset, risk_contrib in risk_attr.asset_risk_contributions.items():
    print(f"  {asset}: {risk_contrib:.2%}")

print("\nMarginal Risk Contributions:")
for asset, marginal in risk_attr.marginal_risk_contributions.items():
    print(f"  {asset}: {marginal:.2%}")

print("\nComponent VaR:")
for asset, cvar in risk_attr.component_var.items():
    print(f"  {asset}: {cvar:.2%}")
```

### Risk Contribution Analysis

**Asset Risk Contribution**: How much each asset contributes to total portfolio risk
```python
# Assets ranked by risk contribution
sorted_risk = sorted(
    risk_attr.asset_risk_contributions.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

print("Assets by Risk Contribution:")
for asset, contrib in sorted_risk:
    pct = (contrib / risk_attr.portfolio_risk) * 100
    print(f"  {asset}: {pct:.1f}% of portfolio risk")
```

**Marginal Risk Contribution**: Expected change in portfolio risk from small change in asset weight
```python
# Identify high marginal risk assets
for asset, marginal in risk_attr.marginal_risk_contributions.items():
    if marginal > risk_attr.portfolio_risk:
        print(f"⚠️  {asset} has high marginal risk: {marginal:.2%}")
```

**Diversification Ratio**: Measures diversification benefit
```python
div_ratio = risk_attr.diversification_ratio
if div_ratio > 1.3:
    print(f"✓ Well diversified (ratio: {div_ratio:.2f})")
elif div_ratio > 1.1:
    print(f"→ Moderately diversified (ratio: {div_ratio:.2f})")
else:
    print(f"⚠️  Poorly diversified (ratio: {div_ratio:.2f})")
```

## Time-Weighted vs Money-Weighted Returns

### Time-Weighted Return (TWR)
Eliminates effect of cash flows, measures pure investment performance:

```python
# Calculate TWR for full period
twr = attributor.calculate_twr()
print(f"Time-Weighted Return: {twr:.2%}")

# TWR for specific period
twr_2023 = attributor.calculate_twr(
    start_date='2023-01-01',
    end_date='2023-12-31'
)
print(f"2023 TWR: {twr_2023:.2%}")
```

### Money-Weighted Return (MWR/IRR)
Accounts for timing and size of cash flows:

```python
# Define cash flows
import pandas as pd

cash_flows = pd.Series({
    pd.Timestamp('2023-03-15'): 10000,   # $10k contribution
    pd.Timestamp('2023-06-01'): -5000,   # $5k withdrawal
    pd.Timestamp('2023-09-10'): 15000,   # $15k contribution
})

# Calculate MWR
mwr = attributor.calculate_mwr(
    cash_flows=cash_flows,
    start_value=100000,
    end_value=150000
)
print(f"Money-Weighted Return (IRR): {mwr:.2%}")
```

**Interpretation**: If MWR > TWR, you timed cash flows well. If MWR < TWR, timing was poor.

## Rolling Attribution

Analyze attribution over time to identify trends and consistency.

### Rolling Analysis

```python
# Calculate rolling attribution (1-year windows)
rolling_attr = attributor.rolling_attribution(
    window=252,  # 252 trading days = 1 year
    step=21      # Monthly steps
)

# Analyze results
print("\n=== Rolling Attribution Summary ===")
print(rolling_attr.describe())

# Plot rolling effects
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

rolling_attr['Allocation Effect'].plot(ax=axes[0], title='Allocation Effect Over Time')
rolling_attr['Selection Effect'].plot(ax=axes[1], title='Selection Effect Over Time')
rolling_attr['Active Return'].plot(ax=axes[2], title='Active Return Over Time')

plt.tight_layout()
plt.show()
```

### Consistency Analysis

```python
# Analyze consistency of attribution effects
allocation_positive = (rolling_attr['Allocation Effect'] > 0).sum()
selection_positive = (rolling_attr['Selection Effect'] > 0).sum()
total_periods = len(rolling_attr)

print(f"\nConsistency Analysis:")
print(f"Allocation positive: {allocation_positive/total_periods:.1%} of periods")
print(f"Selection positive: {selection_positive/total_periods:.1%} of periods")

# Average effects
print(f"\nAverage Effects:")
print(f"Allocation: {rolling_attr['Allocation Effect'].mean():.2%}")
print(f"Selection: {rolling_attr['Selection Effect'].mean():.2%}")
```

## Risk-Adjusted Performance Metrics

### Sharpe Ratio
```python
sharpe = attributor.calculate_sharpe_ratio(risk_free_rate=0.02)
print(f"Sharpe Ratio: {sharpe:.2f}")
```

**Interpretation**: 
- < 1: Poor risk-adjusted performance
- 1-2: Good performance
- > 2: Excellent performance

### Information Ratio
```python
ir = attributor.calculate_information_ratio()
print(f"Information Ratio: {ir:.2f}")
```

**Interpretation**: Measures active return per unit of tracking error
- < 0.5: Poor active management
- 0.5-0.75: Decent active management
- > 0.75: Strong active management

### Sortino Ratio
```python
sortino = attributor.calculate_sortino_ratio(risk_free_rate=0.02)
print(f"Sortino Ratio: {sortino:.2f}")
```

**Interpretation**: Like Sharpe but only penalizes downside volatility

## Interactive Dashboard

### Launching the Dashboard

```bash
# Start the Attribution Dashboard
streamlit run src/dashboard/attribution_dashboard.py
```

### Dashboard Features

1. **Performance Attribution Tab**
   - Key performance metrics
   - Brinson attribution breakdown
   - Active return decomposition
   - Asset contribution waterfall chart
   - Factor attribution analysis
   - Detailed attribution table

2. **Risk Attribution Tab**
   - Portfolio risk metrics
   - Risk contribution charts
   - Component VaR analysis
   - Marginal risk contributions
   - Diversification metrics

3. **Rolling Analysis Tab**
   - Configurable rolling windows
   - Time-series attribution effects
   - Consistency analysis
   - Summary statistics

4. **Reports Tab**
   - Full text reports
   - Downloadable reports
   - Monthly asset contributions
   - Time-series visualizations

## API Reference

### PerformanceAttributor Class

#### Initialization
```python
PerformanceAttributor(
    returns: pd.DataFrame,
    portfolio_weights: Dict[str, float],
    benchmark_weights: Optional[Dict[str, float]] = None,
    prices: Optional[pd.DataFrame] = None
)
```

#### Key Methods

**brinson_attribution(start_date=None, end_date=None) -> AttributionResult**
Calculate Brinson-Fachler attribution.

**factor_attribution(factors=None) -> Dict[str, float]**
Perform factor-based attribution analysis.

**calculate_asset_contributions(period='full') -> pd.DataFrame**
Calculate asset contributions (periods: 'full', 'daily', 'monthly', 'yearly').

**risk_attribution() -> RiskAttribution**
Perform risk attribution analysis.

**calculate_twr(start_date=None, end_date=None) -> float**
Calculate time-weighted return.

**calculate_mwr(cash_flows, start_value, end_value) -> float**
Calculate money-weighted return (IRR).

**rolling_attribution(window=252, step=21) -> pd.DataFrame**
Calculate rolling attribution over time.

**calculate_sharpe_ratio(risk_free_rate=0.02) -> float**
Calculate Sharpe ratio.

**calculate_information_ratio() -> float**
Calculate information ratio.

**calculate_sortino_ratio(risk_free_rate=0.02) -> float**
Calculate Sortino ratio.

**generate_attribution_report() -> str**
Generate comprehensive text report.

## Best Practices

1. **Regular Review**: Analyze attribution monthly or quarterly
2. **Benchmark Selection**: Choose appropriate benchmark for comparison
3. **Factor Analysis**: Understand factor exposures and adjust as needed
4. **Risk Balance**: Monitor risk contributions alongside return contributions
5. **Consistency**: Track attribution over time to identify systematic patterns
6. **Documentation**: Keep records of major allocation decisions and rationale

## Complete Example Workflow

```python
from src.portfolio.performance_attribution import PerformanceAttributor
import pandas as pd

# 1. Load data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# 2. Define weights
portfolio_weights = {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1}
benchmark_weights = {'BTC': 0.25, 'ETH': 0.25, 'SOL': 0.25, 'AVAX': 0.25}

# 3. Initialize
attr = PerformanceAttributor(returns, portfolio_weights, benchmark_weights)

# 4. Brinson attribution
attribution = attr.brinson_attribution()
print(f"Active Return: {attribution.active_return:.2%}")
print(f"Allocation: {attribution.allocation_effect:.2%}")
print(f"Selection: {attribution.selection_effect:.2%}")

# 5. Factor attribution
factor_contrib = attr.factor_attribution()

# 6. Risk attribution
risk_attr = attr.risk_attribution()
print(f"Portfolio Risk: {risk_attr.portfolio_risk:.2%}")

# 7. Asset contributions
monthly_contrib = attr.calculate_asset_contributions('monthly')

# 8. Risk-adjusted metrics
sharpe = attr.calculate_sharpe_ratio()
ir = attr.calculate_information_ratio()
print(f"Sharpe: {sharpe:.2f}, IR: {ir:.2f}")

# 9. Generate report
report = attr.generate_attribution_report()
with open('attribution_report.txt', 'w') as f:
    f.write(report)
```

## Troubleshooting

### Common Issues

**Issue**: Active return doesn't match allocation + selection + interaction
- **Solution**: This is expected in Brinson attribution; effects are approximate

**Issue**: Risk contributions don't sum to portfolio risk
- **Solution**: Check that weights sum to 1.0 and data has no NaN values

**Issue**: Information Ratio is very high/low
- **Solution**: Verify benchmark weights are appropriate; high IR may indicate benchmark mismatch

## References

- [Brinson Attribution Model](https://www.investopedia.com/terms/b/brinson-model.asp)
- [Factor Attribution](https://www.cfainstitute.org/-/media/documents/article/rf-brief/factor-based-attribution.ashx)
- [Performance Attribution Methods](https://www.factset.com/hubfs/Resources/White%20Papers/performance-attribution-methods.pdf)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/TradingSystemStack/issues
- Documentation: docs/portfolio/
