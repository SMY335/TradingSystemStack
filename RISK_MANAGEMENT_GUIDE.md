# ⚠️ Risk Management Guide

Complete guide for advanced portfolio risk analysis and management.

---

## 🎯 Overview

The Risk Management module provides professional-grade risk analysis tools:

- **VaR & CVaR** - Value at Risk and Conditional VaR analysis
- **Monte Carlo Simulation** - 10,000+ scenario simulation
- **Stress Testing** - Market crash and volatility scenarios
- **Risk Attribution** - Component VaR and marginal risk
- **Interactive Dashboard** - Real-time risk visualization

---

## 🚀 Quick Start

### 1. Launch Risk Dashboard

```bash
./run_risk_dashboard.sh
```

Open your browser → `http://localhost:8504`

### 2. Programmatic Usage

```python
from src.portfolio.risk_manager import RiskManager
import pandas as pd

# Historical returns data
returns_df = pd.DataFrame({
    'BTC/USDT': [...],
    'ETH/USDT': [...],
    'SOL/USDT': [...]
})

# Portfolio weights
weights = {'BTC/USDT': 0.5, 'ETH/USDT': 0.3, 'SOL/USDT': 0.2}

# Create Risk Manager
rm = RiskManager(returns_df, weights, risk_free_rate=0.02)

# Calculate risk metrics
metrics = rm.calculate_risk_metrics()

print(f"VaR 95%: {metrics.var_95:.2%}")
print(f"CVaR 95%: {metrics.cvar_95:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

---

## 📊 Risk Metrics Explained

### Value at Risk (VaR)

**Definition:** Maximum expected loss at a given confidence level

- **VaR 95%**: Loss exceeded 5% of the time (1 in 20 days)
- **VaR 99%**: Loss exceeded 1% of the time (1 in 100 days)

**Example:**
- VaR 95% = -3.5%
- Interpretation: On 95% of days, losses will not exceed 3.5%

**Calculation Methods:**

1. **Historical VaR** (default)
```python
var_95 = rm.calculate_var(method='historical', confidence=0.95)
```

2. **Parametric VaR** (assumes normal distribution)
```python
var_95 = rm.calculate_var(method='parametric', confidence=0.95)
```

3. **Monte Carlo VaR** (simulation-based)
```python
var_95 = rm.calculate_var(method='monte_carlo', confidence=0.95, n_scenarios=10000)
```

### Conditional VaR (CVaR)

**Definition:** Expected loss when VaR threshold is exceeded

Also known as **Expected Shortfall** or **Tail VaR**

- **CVaR 95%**: Average of worst 5% of returns
- **CVaR 99%**: Average of worst 1% of returns

```python
cvar_95 = rm.calculate_cvar(confidence=0.95)
```

**Why CVaR is better than VaR:**
- Accounts for tail risk severity
- Captures "black swan" events
- Subadditive (diversification benefit)

### Maximum Drawdown

**Definition:** Largest peak-to-trough decline

```python
max_dd = rm.calculate_max_drawdown()
```

**Interpretation:**
- < 20%: Low risk
- 20-40%: Moderate risk
- > 40%: High risk

### Sharpe Ratio

**Definition:** Risk-adjusted return

```
Sharpe = (Return - Risk-Free Rate) / Volatility
```

**Interpretation:**
- < 1: Poor
- 1-2: Good
- 2-3: Very good
- > 3: Excellent

### Sortino Ratio

**Definition:** Like Sharpe, but only penalizes downside volatility

```python
metrics = rm.calculate_risk_metrics()
sortino = metrics.sortino_ratio
```

Better than Sharpe for asymmetric returns.

### Tail Risk Metrics

**Skewness:**
- Negative = more downside risk (bad)
- Positive = more upside potential (good)

**Kurtosis:**
- Positive = fatter tails (more extreme events)
- Negative = thinner tails (fewer extreme events)

```python
print(f"Skewness: {metrics.skewness:.2f}")
print(f"Kurtosis: {metrics.kurtosis:.2f}")
```

---

## 🎲 Monte Carlo Simulation

Simulate 10,000+ scenarios based on historical statistics:

```python
# Run simulation
simulated_returns = rm.monte_carlo_simulation(
    n_scenarios=10000,
    time_horizon=1  # days
)

# Analyze results
import numpy as np

mean = simulated_returns.mean()
p5 = np.percentile(simulated_returns, 5)  # Worst 5%
p95 = np.percentile(simulated_returns, 95)  # Best 5%

print(f"Expected Return: {mean:.2%}")
print(f"5th Percentile: {p5:.2%}")
print(f"95th Percentile: {p95:.2%}")
```

**Use Cases:**
- Risk forecasting
- Scenario analysis
- Confidence intervals
- Probability distributions

---

## 💥 Stress Testing

Test portfolio under extreme market conditions:

```python
stress_results = rm.stress_test()

for result in stress_results:
    print(f"\n{result.scenario_name}:")
    print(f"  Portfolio Loss: {result.portfolio_loss:.2f}%")
    print(f"  Worst Asset: {result.worst_asset}")
    print(f"  Diversification Benefit: {result.diversification_benefit:.2f}%")
```

**Built-in Scenarios:**

1. **Market Crash -20%**
   - All assets drop 20%
   - Tests moderate bear market

2. **Market Crash -50%**
   - Severe market crash
   - Tests extreme scenarios

3. **Volatility Spike 3x**
   - Volatility increases 3x
   - Tests uncertainty periods

4. **Correlation Breakdown**
   - All correlations → 1
   - Tests diversification failure

**Custom Scenarios:**

```python
# Define custom scenario
custom_returns = returns_df - 0.30  # -30% crash
custom_portfolio = (custom_returns * weights).sum(axis=1).mean()

print(f"Custom scenario loss: {custom_portfolio:.2%}")
```

---

## 📈 Risk Attribution

### Component VaR

**Definition:** Each asset's contribution to total portfolio VaR

```python
component_var = rm.component_var(confidence=0.95)

for asset, cvar in component_var.items():
    print(f"{asset}: {cvar:.4f}")

# Component VaRs sum to Portfolio VaR
total_cvar = sum(component_var.values())
portfolio_var = rm.calculate_var('historical', 0.95)
assert abs(total_cvar - portfolio_var) < 1e-6  # Should be equal
```

**Use Cases:**
- Identify high-risk positions
- Optimize risk allocation
- Risk budgeting

### Marginal VaR

**Definition:** Change in portfolio VaR from small increase in position

```python
marginal_var = rm.marginal_var(confidence=0.95)

for asset, mvar in marginal_var.items():
    print(f"{asset}: {mvar:.4f}")
```

**Interpretation:**
- Positive = increasing position increases VaR
- Negative = increasing position decreases VaR (diversifier)

**Use Cases:**
- Rebalancing decisions
- Adding new positions
- Hedging strategies

### Rolling VaR

Track VaR over time:

```python
rolling_var = rm.rolling_var(window=30, confidence=0.95)

# Plot
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rolling_var.index,
    y=rolling_var * 100,
    name='Rolling VaR 95%'
))
fig.show()
```

---

## 🔔 Risk Alerts

Set up automatic risk monitoring:

```python
def monitor_risk(rm, thresholds):
    """Monitor portfolio risk and send alerts"""
    metrics = rm.calculate_risk_metrics()

    alerts = []

    # VaR threshold
    if abs(metrics.var_95) > thresholds['var_95']:
        alerts.append(f"⚠️ VaR 95% exceeded: {metrics.var_95:.2%}")

    # Drawdown threshold
    if abs(metrics.max_drawdown) > thresholds['max_drawdown']:
        alerts.append(f"⚠️ Max Drawdown exceeded: {metrics.max_drawdown:.2%}")

    # Sharpe ratio threshold
    if metrics.sharpe_ratio < thresholds['min_sharpe']:
        alerts.append(f"⚠️ Sharpe Ratio below target: {metrics.sharpe_ratio:.2f}")

    return alerts

# Example usage
thresholds = {
    'var_95': 0.05,  # 5% VaR limit
    'max_drawdown': 0.20,  # 20% max drawdown
    'min_sharpe': 1.0  # Minimum Sharpe ratio
}

alerts = monitor_risk(rm, thresholds)
for alert in alerts:
    print(alert)
    # Send to Telegram, email, etc.
```

---

## 📊 Dashboard Features

### Risk Overview Tab

- Key risk metrics (VaR, CVaR, Sharpe, Sortino)
- Returns distribution histogram
- Rolling VaR chart
- Drawdown analysis
- Component VaR by asset

### Monte Carlo Tab

- 10,000 scenario simulation
- Distribution visualization
- Percentile analysis
- Confidence intervals

### Stress Tests Tab

- Pre-defined scenarios
- Portfolio impact visualization
- Worst asset identification
- Diversification analysis

### Advanced Metrics Tab

- Correlation heatmap
- Marginal VaR analysis
- Tail risk metrics
- Full risk report

---

## 🎓 Best Practices

### 1. Regular Monitoring

```python
# Daily risk check
def daily_risk_check():
    rm = RiskManager(recent_returns, weights)
    metrics = rm.calculate_risk_metrics()

    # Log metrics
    log_to_database(metrics)

    # Check thresholds
    if check_risk_limits(metrics):
        send_alert()
```

### 2. Multiple Confidence Levels

Don't rely on single VaR level:

```python
var_90 = rm.calculate_var('historical', 0.90)
var_95 = rm.calculate_var('historical', 0.95)
var_99 = rm.calculate_var('historical', 0.99)

print(f"VaR 90%: {var_90:.2%}")
print(f"VaR 95%: {var_95:.2%}")
print(f"VaR 99%: {var_99:.2%}")
```

### 3. Compare VaR Methods

```python
historical = rm.calculate_var('historical', 0.95)
parametric = rm.calculate_var('parametric', 0.95)
monte_carlo = rm.calculate_var('monte_carlo', 0.95)

print(f"Historical VaR: {historical:.2%}")
print(f"Parametric VaR: {parametric:.2%}")
print(f"Monte Carlo VaR: {monte_carlo:.2%}")
```

### 4. Backtest VaR

Verify VaR accuracy:

```python
# Count VaR breaches
var_95 = rm.calculate_var('historical', 0.95)
breaches = (rm.portfolio_returns < var_95).sum()
total_days = len(rm.portfolio_returns)
breach_rate = breaches / total_days

print(f"VaR 95% breach rate: {breach_rate:.2%}")
print(f"Expected: 5.00%")
print(f"Difference: {abs(breach_rate - 0.05):.2%}")
```

### 5. Use CVaR for Decision Making

CVaR is better than VaR for:
- Portfolio optimization
- Risk limits
- Capital allocation

```python
# Optimize for CVaR instead of VaR
from src.portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
weights = optimizer.optimize(
    returns_df,
    method='MIN_CVAR',  # Minimize CVaR
    risk_measure='CVaR'
)
```

---

## ⚠️ Limitations

### 1. Historical Dependency

All metrics based on historical data:
- Past ≠ Future
- Regime changes not captured
- Black swans underestimated

**Solution:** Combine with stress testing and scenario analysis

### 2. Normal Distribution Assumption

Parametric VaR assumes normal returns:
- Real returns have fat tails
- Skewed distributions
- Extreme events more common

**Solution:** Use historical or Monte Carlo VaR

### 3. Correlation Instability

Correlations change over time:
- Increase during crises
- Diversification fails when needed most

**Solution:** Test correlation breakdown scenarios

### 4. Liquidity Not Considered

VaR assumes you can exit positions:
- Crypto markets can freeze
- Slippage in volatile markets
- Gap risk

**Solution:** Add liquidity buffers and wider stop losses

---

## 🔗 Integration

### With Portfolio Optimization

```python
from src.portfolio.optimizer import PortfolioOptimizer

# Optimize with risk constraint
optimizer = PortfolioOptimizer()
weights = optimizer.optimize(
    returns_df,
    method='MAX_SHARPE',
    constraints={'max_risk': 0.15}  # Max 15% volatility
)

# Verify risk
rm = RiskManager(returns_df, weights)
metrics = rm.calculate_risk_metrics()
print(f"Portfolio Risk: {metrics.volatility:.2%}")
```

### With Paper Trading

```python
from src.paper_trading.live_bot import LiveBot

# Check risk before trades
def check_risk_before_trade(position, current_portfolio):
    # Simulate new portfolio
    new_weights = calculate_new_weights(position, current_portfolio)

    # Check risk
    rm = RiskManager(recent_returns, new_weights)
    var_95 = rm.calculate_var('historical', 0.95)

    # Risk limit
    if abs(var_95) > 0.05:  # 5% max VaR
        return False, "VaR limit exceeded"

    return True, "Risk check passed"
```

---

## 📚 Additional Resources

- [Value at Risk (Wikipedia)](https://en.wikipedia.org/wiki/Value_at_risk)
- [Expected Shortfall (Investopedia)](https://www.investopedia.com/terms/c/conditional_value_at_risk.asp)
- [Risk Management (CFA Institute)](https://www.cfainstitute.org/)

---

## 🎉 Next Steps

1. **Launch Dashboard:** `./run_risk_dashboard.sh`
2. **Explore Metrics:** Try different portfolios and assets
3. **Run Stress Tests:** Test worst-case scenarios
4. **Set Alerts:** Monitor risk in real-time
5. **Integrate:** Use in your trading workflow

---

**Stay Risk-Aware! ⚠️**

*Part of TradingSystemStack - Phase 5: Risk & Attribution*
