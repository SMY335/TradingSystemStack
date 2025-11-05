# 📊 Performance Attribution Guide

Complete guide for portfolio performance attribution analysis.

---

## 🎯 Overview

The Performance Attribution module provides professional attribution analysis:

- **Brinson Attribution** - Allocation vs Selection effects
- **Risk Attribution** - Risk contribution by asset
- **Factor Attribution** - Factor-based return decomposition
- **Rolling Analysis** - Time-series attribution
- **Interactive Dashboard** - Real-time visualization

---

## 🚀 Quick Start

### 1. Launch Attribution Dashboard

```bash
./run_attribution_dashboard.sh
```

Open your browser → `http://localhost:8505`

### 2. Programmatic Usage

```python
from src.portfolio.performance_attribution import PerformanceAttributor
import pandas as pd

# Historical returns
returns_df = pd.DataFrame({
    'BTC/USDT': [...],
    'ETH/USDT': [...],
    'SOL/USDT': [...]
})

# Portfolio weights
weights = {'BTC/USDT': 0.5, 'ETH/USDT': 0.3, 'SOL/USDT': 0.2}

# Benchmark weights (optional, defaults to equal weight)
benchmark_weights = {'BTC/USDT': 0.33, 'ETH/USDT': 0.33, 'SOL/USDT': 0.34}

# Create Attributor
attr = PerformanceAttributor(
    returns_df,
    weights,
    benchmark_weights=benchmark_weights
)

# Brinson attribution
attribution = attr.brinson_attribution()
print(f"Active Return: {attribution.active_return:.2%}")
print(f"Allocation Effect: {attribution.allocation_effect:.2%}")
print(f"Selection Effect: {attribution.selection_effect:.2%}")
```

---

## 📊 Attribution Methods

### 1. Brinson Attribution

**Brinson-Fachler Model** - Industry standard for attribution

Decomposes **Active Return** into three components:

```
Active Return = Portfolio Return - Benchmark Return
             = Allocation + Selection + Interaction
```

#### Components:

**Allocation Effect:**
- Returns from over/underweight decisions
- "Did we have more/less than benchmark?"
- Strategic asset allocation skill

**Selection Effect:**
- Returns from asset picking within sectors
- "Did our assets outperform?"
- Tactical selection skill

**Interaction Effect:**
- Combined effect of allocation + selection
- Overweight best performers = positive
- Overweight worst performers = negative

#### Example:

```python
brinson = attr.brinson_attribution()

print(f"Portfolio Return:  {brinson.total_return:.2%}")
print(f"Benchmark Return:  {brinson.benchmark_return:.2%}")
print(f"Active Return:     {brinson.active_return:.2%}")
print(f"")
print(f"Allocation Effect: {brinson.allocation_effect:.2%}")
print(f"Selection Effect:  {brinson.selection_effect:.2%}")
print(f"Interaction:       {brinson.interaction_effect:.2%}")
```

**Output:**
```
Portfolio Return:  15.2%
Benchmark Return:  12.5%
Active Return:     2.7%

Allocation Effect: +1.2%  ← Good overweight decisions
Selection Effect:  +1.8%  ← Good asset picking
Interaction:       -0.3%  ← Overweight underperformers
```

#### Asset-Level Attribution:

```python
for asset, contrib in brinson.asset_contributions.items():
    print(f"\n{asset}:")
    print(f"  Allocation:  {contrib['allocation']:+.2%}")
    print(f"  Selection:   {contrib['selection']:+.2%}")
    print(f"  Total:       {contrib['total']:+.2%}")
```

### 2. Risk Attribution

**Question:** Which assets contribute most to portfolio risk?

```python
risk_attr = attr.risk_attribution()

print(f"Portfolio Risk: {risk_attr.portfolio_risk:.2%}")
print(f"Diversification Ratio: {risk_attr.diversification_ratio:.2f}")

# Risk contribution
for asset, pct in risk_attr.risk_contribution_pct.items():
    print(f"{asset}: {pct:.1%} of portfolio risk")
```

**Metrics:**

- **Component Risk**: Absolute risk contribution
- **Marginal Risk**: Change in risk from position increase
- **Risk Contribution %**: Percentage of total risk
- **Diversification Ratio**: >1 = diversification benefit

**Example:**
```
Portfolio Risk: 25.3%
Diversification Ratio: 1.15  ← 15% diversification benefit

Risk Contribution:
  BTC/USDT: 45%  ← Largest risk contributor
  ETH/USDT: 35%
  SOL/USDT: 20%
```

### 3. Factor Attribution

**Question:** What factors drive portfolio returns?

Decomposes returns into common factors:

```python
factor_attr = attr.factor_attribution()

for factor, contribution in factor_attr.items():
    print(f"{factor}: {contribution:+.2%}")
```

**Built-in Factors:**

1. **Market Factor** - Overall market movement
2. **Value Factor** - Value spread between assets
3. **Momentum Factor** - Momentum effect
4. **Alpha** - Unexplained (skill-based) returns

**Example:**
```
Market:    +8.5%  ← Market beta exposure
Value:     +1.2%  ← Value tilt
Momentum:  +2.3%  ← Momentum capture
Alpha:     +3.0%  ← Manager skill
```

**Custom Factors:**

```python
# Define custom factors
import pandas as pd

custom_factors = pd.DataFrame({
    'Market': market_returns,
    'Size': size_premium,
    'Quality': quality_factor,
    'Volatility': vol_factor
}, index=returns_df.index)

# Run attribution
factor_attr = attr.factor_attribution(factors=custom_factors)
```

### 4. Rolling Attribution

Track attribution over time:

```python
rolling_attr = attr.rolling_attribution(window=30)

print(rolling_attr.head())
```

**Output:**
```
date        portfolio_return  benchmark_return  active_return  allocation  selection
2023-01-30       0.052            0.045            0.007         0.003      0.004
2023-01-31       0.048            0.043            0.005         0.002      0.003
...
```

**Use Cases:**
- Identify skill consistency
- Detect strategy changes
- Time-series analysis

---

## 📈 Performance Metrics

### Time-Weighted Return (TWR)

**Definition:** Geometric average return, independent of cash flows

```python
twr = attr.time_weighted_return()
print(f"TWR: {twr:.2%}")
```

**Best for:**
- Comparing managers
- Measuring skill
- Standard performance reporting

### Money-Weighted Return (MWR/IRR)

**Definition:** Internal Rate of Return, accounts for cash flow timing

```python
# With cash flows
cash_flows = pd.Series({
    '2023-01-01': 10000,   # Initial investment
    '2023-06-01': 5000,    # Additional investment
    '2023-12-31': -18000   # Withdrawal
})

mwr = attr.money_weighted_return(cash_flows)
print(f"MWR: {mwr:.2%}")
```

**Best for:**
- Investor returns
- Cash flow impact
- Personal performance

**TWR vs MWR:**

| Scenario | TWR | MWR |
|----------|-----|-----|
| No cash flows | Same | Same |
| Invest before rally | Same | Higher |
| Invest after rally | Same | Lower |

---

## 🎓 Interpreting Results

### Positive Allocation Effect

**Meaning:** Good overweight/underweight decisions

**Examples:**
- Overweight outperformers (BTC +20%, weight 50% vs 33%)
- Underweight underperformers (ADA -10%, weight 5% vs 20%)

**Action:** Continue strategic allocation

### Positive Selection Effect

**Meaning:** Good asset picking

**Examples:**
- Portfolio assets outperformed benchmark
- Timing of trades was good
- Better execution

**Action:** Continue tactical decisions

### Negative Interaction

**Meaning:** Overweight wrong assets

**Examples:**
- Overweight underperformers
- Underweight outperformers

**Action:** Review allocation process

### High Risk Concentration

**Meaning:** Few assets drive most risk

**Examples:**
- BTC = 60% of portfolio risk
- Top 2 assets = 80% of risk

**Action:**
- Consider diversification
- Or accept concentration if intentional

---

## 🔍 Advanced Analysis

### Performance Attribution Report

Generate comprehensive report:

```python
from src.portfolio.performance_attribution import generate_attribution_report

report = generate_attribution_report(attr)
print(report)
```

### Asset Contribution to Return

Calculate each asset's total contribution:

```python
contributions = attr.asset_contribution_to_return()

for asset, contrib in contributions.items():
    print(f"{asset}: {contrib:.2%}")
```

### Complete Summary

Get all analysis in one call:

```python
summary = attr.generate_attribution_summary()

# Access components
brinson = summary['brinson_attribution']
risk_attr = summary['risk_attribution']
metrics = summary['performance_metrics']
factors = summary['factor_attribution']
```

---

## 📊 Dashboard Features

### Attribution Overview Tab

- Key performance metrics
- Brinson attribution waterfall
- Portfolio vs Benchmark comparison
- Cumulative returns chart

### Asset Analysis Tab

- Asset-level attribution
- Detailed contribution table
- Risk attribution pie chart
- Component risk breakdown

### Rolling Analysis Tab

- Rolling attribution charts
- Time-series decomposition
- Trend analysis
- Statistics summary

### Factor Attribution Tab

- Factor contribution chart
- Factor breakdown table
- TWR vs MWR comparison
- Full attribution report

---

## 🎯 Use Cases

### 1. Manager Evaluation

Compare portfolio manager performance:

```python
def evaluate_manager(returns_df, weights, benchmark_weights):
    attr = PerformanceAttributor(returns_df, weights, benchmark_weights)

    brinson = attr.brinson_attribution()
    metrics = attr.calculate_performance_metrics()

    # Scoring
    score = 0
    score += 20 if brinson.active_return > 0 else 0  # Beat benchmark
    score += 20 if brinson.allocation_effect > 0 else 0  # Good allocation
    score += 20 if brinson.selection_effect > 0 else 0  # Good selection
    score += 20 if metrics.sharpe_ratio > 1.5 else 0  # Good risk-adjusted
    score += 20 if metrics.information_ratio > 0.5 else 0  # Consistent alpha

    return score, brinson, metrics

score, brinson, metrics = evaluate_manager(returns_df, weights, benchmark)
print(f"Manager Score: {score}/100")
```

### 2. Strategy Analysis

Identify what's working:

```python
def analyze_strategy(attr):
    brinson = attr.brinson_attribution()

    if brinson.allocation_effect > brinson.selection_effect:
        print("✅ Allocation skill dominates")
        print("→ Focus on strategic allocation")
    else:
        print("✅ Selection skill dominates")
        print("→ Focus on tactical trading")

    # Risk analysis
    risk_attr = attr.risk_attribution()
    max_contributor = max(
        risk_attr.risk_contribution_pct,
        key=risk_attr.risk_contribution_pct.get
    )

    print(f"\n⚠️ Largest risk: {max_contributor}")
```

### 3. Rebalancing Decisions

Identify opportunities:

```python
def suggest_rebalancing(attr):
    suggestions = []

    brinson = attr.brinson_attribution()

    # Check asset contributions
    for asset, contrib in brinson.asset_contributions.items():
        # Negative allocation effect = wrong weight
        if contrib['allocation'] < -0.01:
            suggestions.append(f"Rebalance {asset} weight")

        # Negative selection = underperforming
        if contrib['selection'] < -0.01:
            suggestions.append(f"Review {asset} selection")

    return suggestions
```

### 4. Risk Budgeting

Allocate risk intentionally:

```python
def risk_budget_check(attr, target_contributions):
    risk_attr = attr.risk_attribution()

    for asset, target in target_contributions.items():
        actual = risk_attr.risk_contribution_pct[asset]
        diff = actual - target

        if abs(diff) > 0.05:  # 5% tolerance
            print(f"⚠️ {asset}: {actual:.1%} vs target {target:.1%}")
```

---

## 📚 Best Practices

### 1. Regular Attribution

Run attribution monthly/quarterly:

```python
def monthly_attribution_report(month_end_date):
    # Get month data
    returns = get_returns_for_month(month_end_date)
    weights = get_weights_at_month_end(month_end_date)

    # Run attribution
    attr = PerformanceAttributor(returns, weights)
    summary = attr.generate_attribution_summary()

    # Save report
    save_to_database(month_end_date, summary)

    # Send report
    send_email_report(summary)
```

### 2. Compare to Benchmark

Always use meaningful benchmark:

```python
# Bad: Random benchmark
benchmark_weights = equal_weight(assets)

# Good: Relevant benchmark
benchmark_weights = market_cap_weight(assets)
# Or: Your strategic allocation
benchmark_weights = strategic_weights
```

### 3. Consider Transaction Costs

Attribution ignores costs:

```python
# Adjust for costs
gross_active_return = brinson.active_return
transaction_costs = calculate_costs(trades)
net_active_return = gross_active_return - transaction_costs

print(f"Gross Active: {gross_active_return:.2%}")
print(f"Costs: {transaction_costs:.2%}")
print(f"Net Active: {net_active_return:.2%}")
```

### 4. Look at Risk-Adjusted Returns

Don't just chase returns:

```python
metrics = attr.calculate_performance_metrics()

# Return per unit of risk
return_per_risk = metrics.annualized_return / metrics.volatility

# Sharpe ratio
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")

# Information ratio (vs benchmark)
print(f"IR: {metrics.information_ratio:.2f}")
```

---

## ⚠️ Limitations

### 1. Benchmark Dependency

Attribution quality depends on benchmark:
- Use relevant benchmark
- Consider multiple benchmarks
- Custom benchmarks for strategies

### 2. Time Period Sensitivity

Short periods = noisy attribution:
- Use longer periods (90+ days)
- Rolling analysis for stability
- Multiple time horizons

### 3. Factor Model Assumptions

Factor attribution assumes:
- Linear factor relationships
- Stable factor loadings
- Factor independence

**Solution:** Use multiple models, validate assumptions

### 4. Transaction Costs Ignored

Attribution is gross of costs:
- Track costs separately
- Adjust returns for costs
- Consider net attribution

---

## 🔗 Integration

### With Risk Management

```python
from src.portfolio.risk_manager import RiskManager

# Combined analysis
def comprehensive_analysis(returns_df, weights, benchmark_weights):
    # Attribution
    attr = PerformanceAttributor(returns_df, weights, benchmark_weights)
    brinson = attr.brinson_attribution()

    # Risk
    rm = RiskManager(returns_df, weights)
    risk_metrics = rm.calculate_risk_metrics()

    # Report
    print(f"Active Return: {brinson.active_return:.2%}")
    print(f"VaR 95%: {risk_metrics.var_95:.2%}")
    print(f"Sharpe: {risk_metrics.sharpe_ratio:.2f}")
```

### With Portfolio Optimization

```python
# Optimize for allocation skill
if brinson.allocation_effect > brinson.selection_effect:
    # Focus on strategic weights
    optimizer.optimize(returns_df, method='MAX_SHARPE')
else:
    # Focus on tactical trades
    run_tactical_strategy()
```

---

## 📖 Additional Resources

- [Brinson Attribution (CFA)](https://www.cfainstitute.org/)
- [Performance Attribution (Investopedia)](https://www.investopedia.com/terms/p/performance-attribution.asp)
- [Factor Models (Papers)](https://papers.ssrn.com/)

---

## 🎉 Next Steps

1. **Launch Dashboard:** `./run_attribution_dashboard.sh`
2. **Run Attribution:** Analyze your portfolio
3. **Interpret Results:** Understand drivers of performance
4. **Take Action:** Improve allocation and selection
5. **Monitor:** Track attribution over time

---

**Understand Your Performance! 📊**

*Part of TradingSystemStack - Phase 5: Risk & Attribution*
