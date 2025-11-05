# 💼 Portfolio Management Guide

Complete guide for using the Portfolio Management module with Riskfolio-Lib.

---

## 🎯 Overview

The Portfolio Management module provides professional-grade portfolio optimization and risk management capabilities:

- **Multi-Asset Portfolios** - Manage crypto, stocks, forex, commodities
- **Portfolio Optimization** - Mean-variance, Risk Parity, Max Sharpe, Min Volatility
- **Automatic Rebalancing** - Time-based or drift-based rebalancing
- **Risk Analytics** - VaR, CVaR, Sharpe, Sortino, Max Drawdown
- **Interactive Dashboard** - Full-featured Streamlit interface

---

## 🚀 Quick Start

### 1. Launch Portfolio Dashboard

```bash
./run_portfolio_dashboard.sh
```

Open your browser → `http://localhost:8503`

### 2. Programmatic Usage

```python
from portfolio import Portfolio, PortfolioOptimizer, PortfolioManager
from portfolio.models import Asset, Position, AssetType
from datetime import datetime

# Create portfolio
portfolio = Portfolio(
    name="My Portfolio",
    initial_capital=100000,
    cash=20000
)

# Add positions
btc = Asset("BTC/USDT", AssetType.CRYPTO, "binance")
btc_position = Position(
    asset=btc,
    quantity=1.5,
    entry_price=45000,
    entry_date=datetime.now(),
    current_price=47000
)
portfolio.add_position(btc_position)

# Optimize portfolio
optimizer = PortfolioOptimizer()
weights = optimizer.optimize(
    returns_data,
    method=OptimizationMethod.MAX_SHARPE
)

print(f"Optimal weights: {weights}")
```

---

## 📊 Features

### 1. Portfolio Optimization Methods

#### Mean-Variance (Markowitz)
Classic portfolio optimization maximizing return for given risk.

```python
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MEAN_VARIANCE,
    target_return=0.15  # 15% annual return
)
```

#### Risk Parity
Equal risk contribution from all assets.

```python
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.RISK_PARITY
)
```

#### Maximum Sharpe Ratio
Maximize risk-adjusted returns.

```python
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MAX_SHARPE,
    risk_measure=RiskMeasure.MV
)
```

#### Minimum Volatility
Minimize portfolio volatility.

```python
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MIN_VOLATILITY
)
```

### 2. Risk Measures

- **MV** - Standard Deviation (Mean-Variance)
- **MAD** - Mean Absolute Deviation
- **CVaR** - Conditional Value at Risk (Expected Shortfall)
- **CDaR** - Conditional Drawdown at Risk

```python
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MAX_SHARPE,
    risk_measure=RiskMeasure.CVaR  # Use CVaR instead of volatility
)
```

### 3. Automatic Rebalancing

```python
from portfolio.portfolio_manager import PortfolioManager, RebalancingConfig

# Configure rebalancing
config = RebalancingConfig(
    frequency='monthly',  # daily, weekly, monthly, quarterly
    threshold=0.05,  # Rebalance if drift > 5%
    optimization_method=OptimizationMethod.MAX_SHARPE,
    lookback_period=90  # Days of history for optimization
)

# Create manager
manager = PortfolioManager(
    portfolio=my_portfolio,
    data_fetcher=fetch_data_function,
    rebalancing_config=config
)

# Check if rebalancing needed
if manager.should_rebalance():
    result = manager.rebalance()

    print(f"Trades required: {result['trades']}")
    print(f"Expected improvement: {result['improvement']}")
```

### 4. Performance Analytics

```python
# Calculate portfolio metrics
metrics = optimizer.calculate_portfolio_metrics(weights, returns_df)

print(f"Expected Return: {metrics['expected_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"VaR (95%): {metrics['var_95']:.2%}")
print(f"CVaR (95%): {metrics['cvar_95']:.2%}")
```

### 5. Efficient Frontier

Calculate and visualize the efficient frontier:

```python
risks, returns, weights_list = optimizer.efficient_frontier(
    returns_df,
    risk_measure=RiskMeasure.MV,
    points=20
)

# Plot efficient frontier
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=risks * 100,
    y=returns * 100,
    mode='markers+lines',
    name='Efficient Frontier'
))
fig.update_layout(
    title='Efficient Frontier',
    xaxis_title='Risk (Volatility %)',
    yaxis_title='Expected Return %'
)
fig.show()
```

---

## 📈 Dashboard Features

### Overview Tab
- Portfolio composition (pie chart)
- Key metrics (value, return, P&L)
- Positions detail table
- Real-time weight allocation

### Optimization Tab
- Multiple optimization methods
- Risk measure selection
- Historical lookback period
- Before/after comparison
- Optimal weights visualization

### Rebalancing Tab
- Frequency configuration
- Drift threshold setting
- Trade calculation
- Expected improvement metrics
- Action recommendations (BUY/SELL)

### Analytics Tab
- Comprehensive performance metrics
- Risk-adjusted returns
- Drawdown analysis
- Value at Risk (VaR/CVaR)
- Historical performance

---

## 🎓 Examples

### Example 1: Simple Portfolio Optimization

```python
import pandas as pd
from portfolio import PortfolioOptimizer, OptimizationMethod

# Historical returns data
returns_df = pd.DataFrame({
    'BTC/USDT': [...],
    'ETH/USDT': [...],
    'BNB/USDT': [...]
})

# Optimize for maximum Sharpe ratio
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MAX_SHARPE
)

print("Optimal Allocation:")
for symbol, weight in weights.items():
    print(f"  {symbol}: {weight*100:.2f}%")
```

### Example 2: Portfolio with Constraints

```python
# Add constraints
constraints = {
    'min_weight': 0.05,  # Minimum 5% per asset
    'max_weight': 0.40,  # Maximum 40% per asset
    'budget': 1.0  # Fully invested
}

weights = optimizer.optimize(
    returns_df,
    method=OptimizationMethod.MAX_SHARPE,
    constraints=constraints
)
```

### Example 3: Automatic Rebalancing System

```python
def fetch_data(symbols, days):
    # Fetch historical data
    # ... implementation
    return returns_df

# Create portfolio manager
manager = PortfolioManager(
    portfolio=my_portfolio,
    data_fetcher=fetch_data,
    rebalancing_config=RebalancingConfig(
        frequency='weekly',
        threshold=0.05
    )
)

# Run periodic check (e.g., daily cron job)
if manager.should_rebalance():
    result = manager.rebalance()

    # Execute trades
    current_prices = get_current_prices()
    execution = manager.execute_trades(
        result['trades'],
        current_prices
    )

    print(f"Executed {execution['executed']} trades")
    print(f"New portfolio value: ${execution['new_total_value']:,.2f}")
```

### Example 4: Compare Multiple Strategies

```python
methods = [
    OptimizationMethod.EQUAL_WEIGHT,
    OptimizationMethod.MIN_VOLATILITY,
    OptimizationMethod.MAX_SHARPE,
    OptimizationMethod.RISK_PARITY
]

results = {}

for method in methods:
    weights = optimizer.optimize(returns_df, method=method)
    metrics = optimizer.calculate_portfolio_metrics(weights, returns_df)

    results[method.value] = {
        'weights': weights,
        'sharpe': metrics['sharpe_ratio'],
        'return': metrics['expected_return'],
        'volatility': metrics['volatility']
    }

# Display comparison
for method, data in results.items():
    print(f"\n{method}:")
    print(f"  Sharpe: {data['sharpe']:.3f}")
    print(f"  Return: {data['return']:.2%}")
    print(f"  Risk: {data['volatility']:.2%}")
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_portfolio.py
```

Tests cover:
- Portfolio data models
- Optimization algorithms
- Efficient frontier calculation
- Portfolio manager functionality
- Trade execution
- Performance analytics

---

## 📊 Performance Metrics Explained

### Sharpe Ratio
Risk-adjusted return measure. Higher is better.
- **< 1**: Sub-optimal
- **1-2**: Good
- **2-3**: Very good
- **> 3**: Excellent

### Sortino Ratio
Like Sharpe but only considers downside risk. Higher is better.

### Max Drawdown
Largest peak-to-trough decline. Lower is better.
- **< 20%**: Low risk
- **20-40%**: Moderate risk
- **> 40%**: High risk

### VaR (Value at Risk)
Maximum expected loss at given confidence level (e.g., 95%).

### CVaR (Conditional VaR)
Expected loss when VaR threshold is exceeded. Measures tail risk.

### Calmar Ratio
Return / Max Drawdown. Higher is better.

---

## ⚙️ Configuration

### Optimizer Configuration

```python
optimizer = PortfolioOptimizer(
    risk_free_rate=0.02,  # 2% annual risk-free rate
    frequency=252,  # Trading days per year
    alpha=0.05  # 5% significance for VaR/CVaR
)
```

### Rebalancing Configuration

```python
config = RebalancingConfig(
    frequency='monthly',  # Rebalancing frequency
    threshold=0.05,  # 5% drift threshold
    optimization_method=OptimizationMethod.MAX_SHARPE,
    risk_measure=RiskMeasure.MV,
    lookback_period=90  # Days of historical data
)
```

---

## 🚨 Important Notes

### 1. Data Requirements
- Minimum 30 days of historical data recommended
- More data = more reliable optimization
- Use consistent timeframes (daily, weekly, etc.)

### 2. Optimization Frequency
- **Too frequent**: High transaction costs
- **Too rare**: Portfolio drift, suboptimal allocation
- **Recommended**: Monthly for most portfolios

### 3. Risk Measures
- **MV (Volatility)**: Symmetric, assumes normal returns
- **CVaR**: Better for fat-tailed distributions
- **CDaR**: Best for measuring crash risk

### 4. Backtesting
Always backtest optimization strategies before live use:

```python
# Backtest optimization strategy
results = backtest_optimization(
    returns_df,
    method=OptimizationMethod.MAX_SHARPE,
    rebalance_frequency='monthly'
)
```

### 5. Transaction Costs
Consider transaction costs in real trading:
- Rebalancing too often = high costs
- Use drift threshold to reduce unnecessary trades
- Group trades when possible

---

## 🔗 Integration

### With Backtesting Module

```python
from backtesting import BacktestEngine
from portfolio import PortfolioOptimizer

# Optimize weights
weights = optimizer.optimize(historical_returns)

# Backtest with optimized weights
engine = BacktestEngine()
results = engine.run(data, weights=weights)
```

### With Paper Trading

```python
from paper_trading import LiveBot
from portfolio import PortfolioManager

# Create manager
manager = PortfolioManager(...)

# Get rebalancing signals
if manager.should_rebalance():
    result = manager.rebalance()

    # Execute via paper trading bot
    bot = LiveBot(...)
    for symbol, weight_change in result['trades'].items():
        bot.place_order(symbol, weight_change)
```

---

## 📚 Additional Resources

- [Riskfolio-Lib Documentation](https://riskfolio-lib.readthedocs.io/)
- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Risk Parity](https://www.investopedia.com/terms/r/risk-parity.asp)

---

## 🎉 Next Steps

1. **Explore Dashboard**: Try different optimization methods
2. **Run Tests**: Validate functionality with `test_portfolio.py`
3. **Backtest Strategies**: Test optimization on historical data
4. **Paper Trade**: Validate with paper trading before live
5. **Monitor Performance**: Track metrics over time

---

**Happy Portfolio Management! 📊💼**

---

*Part of TradingSystemStack - Phase 5: Portfolio & Risk Management*
