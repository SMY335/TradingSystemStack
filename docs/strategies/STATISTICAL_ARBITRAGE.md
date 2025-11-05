# Statistical Arbitrage - Pairs Trading

## Overview

Statistical Arbitrage (Stat Arb) is a quantitative trading strategy that exploits statistical relationships between financial instruments. This implementation focuses on **Pairs Trading**, which trades the spread between two cointegrated assets.

## Core Concept: Cointegration

**What is Cointegration?**

Two assets are cointegrated if their price series move together over time, even though each individual series may be non-stationary. Unlike correlation, cointegration implies a long-term equilibrium relationship.

**Example**: BTC and ETH
- Both are cryptocurrencies
- Both respond to similar market forces
- Their price ratio tends to revert to a mean

## Strategy Mechanics

### The Spread

The spread between two assets is defined as:
```python
spread = price_A - hedge_ratio * price_B
```

Where:
- `price_A`: Price of first asset (e.g., BTC)
- `price_B`: Price of second asset (e.g., ETH)
- `hedge_ratio`: Beta coefficient from linear regression

### Hedge Ratio Calculation

```python
# Method 1: Covariance-based
hedge_ratio = cov(A, B) / var(B)

# Method 2: Linear regression (preferred)
hedge_ratio = polyfit(prices_B, prices_A, deg=1)[0]
```

### Z-Score

The z-score measures how many standard deviations the current spread is from its mean:

```python
z_score = (current_spread - mean_spread) / std_spread
```

## Trading Logic

### Entry Signals

**Long Spread** (when z-score < -2.0):
- Spread is too low
- Buy Asset A
- Sell Asset B (in proportion to hedge ratio)
- Expect mean reversion

**Short Spread** (when z-score > 2.0):
- Spread is too high
- Sell Asset A
- Buy Asset B (in proportion to hedge ratio)
- Expect mean reversion

### Exit Signals

**Mean Reversion Exit** (|z-score| < 0.5):
- Spread has returned to normal
- Close both positions
- Book profits

**Stop Loss** (|z-score| > 3.0):
- Spread continues diverging
- Relationship may have broken
- Close positions to limit losses

## Parameters

```python
lookback: 60              # Window for spread calculation
entry_zscore: 2.0         # Entry threshold
exit_zscore: 0.5          # Exit threshold
stop_loss_zscore: 3.0     # Stop loss threshold
position_size: 1.0        # Position size
recalc_period: 20         # Recalculate hedge ratio every N bars
```

## Implementation Example

```python
from src.quant_strategies.pairs_trading import PairsTradingStrategy
from src.adapters.pairs_adapter import PairsAdapter
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Configure strategy
config = StrategyConfig(
    name="BTC_ETH_Pairs",
    parameters={
        'lookback': 60,
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'stop_loss_zscore': 3.0,
        'position_size': 1.0,
        'recalc_period': 20,
    },
    timeframe="1h",
    symbols=["BTC/USD", "ETH/USD"],
    capital=10000,
    framework=StrategyFramework.BACKTRADER
)

# Create adapter
adapter = PairsAdapter(config)
strategy_class = adapter.to_backtrader()
```

## Selecting Trading Pairs

### Criteria for Good Pairs:

1. **Cointegration Test**
   ```python
   from statsmodels.tsa.stattools import coint
   
   _, p_value, _ = coint(prices_A, prices_B)
   
   # Good pairs have p_value < 0.05
   if p_value < 0.05:
       print("Pairs are cointegrated")
   ```

2. **High Correlation**
   - Correlation > 0.7 preferred
   - Check rolling correlation stability

3. **Similar Market Factors**
   - Same sector/industry
   - Similar market cap
   - Common economic drivers

4. **Liquidity**
   - Both assets must be liquid
   - Low transaction costs
   - Tight bid-ask spreads

### Crypto Pairs Examples:
- BTC/ETH
- ETH/BNB
- Major altcoins during bull markets

### Traditional Pairs Examples:
- XLE/XLF (Energy vs Financials)
- PEP/KO (Pepsi vs Coca-Cola)
- GLD/SLV (Gold vs Silver)

## Risk Management

### Position Sizing

Equal dollar amounts in both legs:
```python
# For long spread:
buy_A_quantity = capital * position_pct / price_A
sell_B_quantity = hedge_ratio * buy_A_quantity
```

### Risks to Monitor

1. **Cointegration Breakdown**
   - Relationship may fail
   - Recalculate periodically
   - Monitor p-value

2. **Execution Risk**
   - Legs may fill at different times
   - Use limit orders
   - Monitor slippage

3. **Tail Risk**
   - Black swan events
   - Use stop losses
   - Don't over-leverage

4. **Transaction Costs**
   - Spreads must be wide enough
   - Account for fees
   - Consider slippage

## Performance Metrics

### Typical Performance (Crypto Pairs):
```
Average Trade Duration: 2-5 days
Win Rate: 65-75%
Profit Factor: 1.5-2.0
Sharpe Ratio: 1.2-1.8
Max Drawdown: -8% to -15%
```

### Key Metrics to Track:
- Spread mean reversion time
- Z-score distribution
- Hedge ratio stability
- Cointegration p-value over time

## Advanced Techniques

### 1. Multiple Pairs Portfolio

Trade multiple cointegrated pairs simultaneously:
```python
pairs = [
    ("BTC", "ETH"),
    ("BTC", "BNB"),
    ("ETH", "BNB"),
]

# Diversification reduces risk
# Increases trade frequency
```

### 2. Dynamic Thresholds

Adjust entry/exit thresholds based on volatility:
```python
entry_threshold = base_threshold * volatility_multiplier
```

### 3. Half-Kelly Position Sizing

Optimize position size using Kelly Criterion:
```python
kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
position_size = kelly_fraction * 0.5  # Half-Kelly for safety
```

## Backtesting Considerations

### Walk-Forward Analysis:
```python
# Train Period: 3 months
# Test Period: 1 month
# Roll forward: Monthly

for start_date in date_range:
    train_data = data[start_date:start_date+90d]
    test_data = data[start_date+90d:start_date+120d]
    
    # Calculate hedge ratio on train
    # Test on test data
```

### Out-of-Sample Testing:
- Never optimize on full dataset
- Reserve 20% for final validation
- Test multiple market regimes

## Common Pitfalls

1. **Over-fitting**
   - Too many parameters
   - Optimized on limited data
   - Solution: Simple models, robust testing

2. **Ignoring Transaction Costs**
   - Can eat all profits
   - Solution: Include realistic costs

3. **Leverage Risk**
   - Amplifies losses
   - Solution: Conservative sizing

4. **Correlation ≠ Cointegration**
   - High correlation doesn't guarantee mean reversion
   - Solution: Always test for cointegration

## Optimization Parameters

```python
param_space = {
    'lookback': [40, 60, 80, 100],
    'entry_zscore': [1.5, 2.0, 2.5, 3.0],
    'exit_zscore': [0.25, 0.5, 0.75, 1.0],
    'stop_loss_zscore': [3.0, 3.5, 4.0],
    'recalc_period': [10, 20, 30],
}
```

### Optimization Tips:
- Use walk-forward analysis
- Test across multiple pairs
- Validate in different market conditions
- Avoid curve-fitting

## Market Conditions

### Works Best In:
- Stable, range-bound markets
- Low volatility environments
- Established relationships

### Struggles In:
- Trending markets (one-directional)
- High volatility periods
- Regime changes
- News-driven moves

## Further Reading

- **Books**:
  - "Algorithmic Trading" by Ernie Chan
  - "Quantitative Trading" by Ernie Chan
  - "Trading and Exchanges" by Larry Harris

- **Papers**:
  - "Pairs Trading: Performance of a Relative Value Arbitrage Rule" (Gatev et al.)
  - "Statistical Arbitrage in the U.S. Equities Market" (Avellaneda & Lee)

- **Tools**:
  - statsmodels (Python cointegration tests)
  - pandas (data analysis)
  - backtrader (backtesting)

## Code Structure

```
src/quant_strategies/
├── pairs_trading.py        # Main strategy
├── cointegration.py       # (Future) Coint tests
└── portfolio_pairs.py     # (Future) Multi-pair

src/adapters/
└── pairs_adapter.py       # Framework adapter
```

## Warnings

⚠️ **Important**:
- Statistical relationships can break down
- Past cointegration doesn't guarantee future
- Always use stop losses
- Monitor cointegration continuously
- Start with paper trading
- Understand the assets you're trading
