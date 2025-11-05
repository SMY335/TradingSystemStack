# Market Making Strategy

## Overview

Market Making is a trading strategy that provides liquidity to markets by continuously placing both buy (bid) and sell (ask) orders. The goal is to profit from the bid-ask spread while managing inventory risk.

## Core Concepts

### Bid-Ask Spread

The spread is the difference between the bid (buy) price and ask (sell) price:

```
Mid Price = (Best Bid + Best Ask) / 2
Spread = Ask Price - Bid Price
```

### Market Maker Role

- **Provide Liquidity**: Always willing to buy and sell
- **Capture Spread**: Profit from difference between bid/ask
- **Inventory Management**: Balance long/short positions
- **Risk Management**: Adjust prices based on inventory

## Simple Market Making Strategy

### Basic Logic

```python
# Calculate prices
mid_price = current_close_price
spread = mid_price * (spread_bps / 10000)

bid_price = mid_price - spread / 2
ask_price = mid_price + spread / 2

# Place orders
if inventory < max_inventory:
    place_bid(bid_price, order_size)

if inventory > -max_inventory:
    place_ask(ask_price, order_size)
```

### Inventory Skewing

Adjust prices based on current inventory to encourage mean reversion:

```python
inventory_skew = inventory / max_inventory * skew_factor

# When long (positive inventory), decrease prices to encourage selling
bid_price *= (1 - inventory_skew)
ask_price *= (1 + inventory_skew)

# When short (negative inventory), increase prices to encourage buying
```

## Parameters

```python
spread_bps: 20          # Spread in basis points (0.20%)
max_inventory: 10       # Maximum position (long or short)
order_size: 1.0         # Size of each order
skew_factor: 0.001      # Inventory adjustment factor
update_freq: 1          # Update orders every N bars
```

## Implementation Example

```python
from src.market_making.simple_mm import SimpleMarketMaker
from src.adapters.mm_adapter import MarketMakerAdapter
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Configure strategy
config = StrategyConfig(
    name="BTC_MarketMaker",
    parameters={
        'spread_bps': 20,
        'max_inventory': 10,
        'order_size': 1.0,
        'skew_factor': 0.001,
        'update_freq': 1,
    },
    timeframe="1m",
    symbols=["BTC/USD"],
    capital=10000,
    framework=StrategyFramework.BACKTRADER
)

# Create adapter
adapter = MarketMakerAdapter(config)
strategy_class = adapter.to_backtrader()
```

## Profitability Mechanics

### Profit from Spread

```
Buy at:  $50,000 (bid)
Sell at: $50,010 (ask)
Profit:  $10 per round trip
```

### Volume Requirements

```python
# Example calculation
spread = 0.20%  # 20 bps
trades_per_day = 100
avg_trade_size = $1,000

daily_profit = trades_per_day * avg_trade_size * (spread / 100)
            = 100 * $1,000 * 0.002
            = $200

# Minus costs:
- Exchange fees: ~$50
- Slippage: ~$30
Net Profit: ~$120/day
```

## Inventory Risk Management

### The Challenge

Market makers accumulate inventory risk:
- Too much long exposure → vulnerable to price drops
- Too much short exposure → vulnerable to price rises

### Solutions

1. **Inventory Limits**
   ```python
   if inventory >= max_inventory:
       stop_placing_bids()
   
   if inventory <= -max_inventory:
       stop_placing_asks()
   ```

2. **Price Skewing**
   ```python
   # Widen spread on undesired side
   # Tighten spread on desired side
   ```

3. **Hedging**
   ```python
   # Use futures/options to hedge inventory
   if abs(inventory) > hedge_threshold:
       place_hedge_order()
   ```

## Advanced Techniques

### 1. Dynamic Spread Adjustment

Adjust spread based on market conditions:

```python
# Volatility-based
base_spread = 20  # bps
volatility_multiplier = current_volatility / avg_volatility
adjusted_spread = base_spread * volatility_multiplier

# Order book imbalance
bid_depth = sum(buy_orders)
ask_depth = sum(sell_orders)
imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)

if imbalance > 0.2:  # More buyers
    increase_ask_spread()
    decrease_bid_spread()
```

### 2. Multi-Level Quoting

Place multiple orders at different price levels:

```python
levels = [
    {'offset': 0.10%, 'size': 2.0},
    {'offset': 0.20%, 'size': 1.5},
    {'offset': 0.30%, 'size': 1.0},
]

for level in levels:
    bid = mid_price * (1 - level['offset'])
    ask = mid_price * (1 + level['offset'])
    place_orders(bid, ask, level['size'])
```

### 3. Adverse Selection Protection

Avoid trading with informed traders:

```python
# Cancel orders quickly after large trades
if large_trade_detected():
    cancel_all_orders()
    wait(cooldown_period)
    re-quote()

# Widen spread during news events
if high_volatility_period():
    spread *= volatility_multiplier
```

## Risk Factors

### 1. Inventory Risk

**Risk**: Holding too much inventory when price moves against you

**Mitigation**:
- Strict inventory limits
- Dynamic hedging
- Position limits per symbol

### 2. Adverse Selection

**Risk**: Trading with informed traders who know something you don't

**Mitigation**:
- Quick order updates
- Wider spreads during volatility
- Order cancellation on large trades

### 3. Technical Risk

**Risk**: System failures, connectivity issues, bugs

**Mitigation**:
- Redundant systems
- Kill switches
- Position monitors
- Heartbeat checks

### 4. Regulatory Risk

**Risk**: Compliance violations, market manipulation accusations

**Mitigation**:
- Understand regulations
- Maintain audit trails
- Avoid wash trading
- Fair pricing

## Performance Metrics

### Key Metrics

```python
# Profitability
Gross PnL = Total Profit from Spreads
Net PnL = Gross PnL - Costs - Slippage

# Efficiency
Fill Rate = Filled Orders / Total Orders
Win Rate = Profitable Trades / Total Trades
Average Spread Captured = Sum(Spreads) / Trades

# Risk
Max Inventory = Peak Absolute Position
Inventory Duration = Time Spent at Max
Sharpe Ratio = Returns / Volatility
```

### Typical Performance

```
Markets: Liquid crypto pairs (BTC, ETH)
Timeframe: 1-5 minute bars
Spread: 10-30 bps

Metrics:
- Win Rate: 75-85%
- Fill Rate: 40-60%
- Sharpe Ratio: 2.0-3.0
- Max Inventory: ±10 units
- Daily Volume: 50-200 trades
```

## Best Markets for Market Making

### Ideal Characteristics:

1. **High Liquidity**
   - Deep order books
   - Consistent volume
   - Multiple participants

2. **Low Volatility**
   - Predictable price action
   - Rare flash crashes
   - Stable spreads

3. **Good Infrastructure**
   - Fast execution
   - Reliable connectivity
   - Low latency

4. **Favorable Fee Structure**
   - Maker rebates
   - Low transaction costs
   - No hidden fees

### Examples:

**Crypto**:
- BTC/USD, ETH/USD (major pairs)
- High volume altcoins
- Stablecoin pairs (lower risk)

**Traditional**:
- Large cap stocks (AAPL, MSFT, GOOGL)
- Major FX pairs (EUR/USD, USD/JPY)
- Liquid ETFs (SPY, QQQ)

## Transaction Costs

### Cost Components

```python
# Per trade costs:
1. Exchange fees: 0.01-0.10%
2. Spread crossing: variable
3. Slippage: 0.01-0.05%
4. Market impact: depends on size

# Infrastructure costs:
5. Server/hosting: $100-1000/month
6. Data feeds: $50-500/month
7. API access: $0-1000/month
```

### Break-Even Analysis

```python
# Minimum spread to be profitable:
min_spread = (exchange_fee * 2) + slippage + desired_profit

# Example:
exchange_fee = 0.05%  # Per side
slippage = 0.02%
desired_profit = 0.10%

min_spread = (0.05% * 2) + 0.02% + 0.10%
          = 0.22%
          = 22 basis points
```

## Optimization

### Parameter Tuning

```python
param_space = {
    'spread_bps': [10, 15, 20, 25, 30],
    'max_inventory': [5, 10, 15, 20],
    'order_size': [0.5, 1.0, 1.5, 2.0],
    'skew_factor': [0.0005, 0.001, 0.0015, 0.002],
    'update_freq': [1, 2, 5],
}
```

### Objectives to Optimize:

1. **Maximize PnL**: Total profits
2. **Minimize Risk**: Sharpe ratio, max drawdown
3. **Balance Inventory**: Time at neutral position
4. **Fill Rate**: Percentage of orders filled

## Common Mistakes

1. **Spreads Too Tight**
   - Costs eat all profit
   - Adverse selection
   - Solution: Calculate break-even spread

2. **No Inventory Control**
   - Accumulate huge positions
   - Cannot exit cleanly
   - Solution: Strict limits + hedging

3. **Ignoring Volatility**
   - Fixed spreads in all conditions
   - Losses during spikes
   - Solution: Dynamic spread adjustment

4. **Over-Leverage**
   - Amplifies all risks
   - Account blow-up risk
   - Solution: Conservative sizing

## Regulatory Considerations

### Requirements May Include:

1. **Registration**: Broker-dealer license
2. **Capital**: Minimum net capital
3. **Reporting**: Trade reporting obligations
4. **Fair Pricing**: NBBO compliance
5. **Risk Controls**: Pre-trade checks

### Best Practices:

- Consult legal/compliance experts
- Maintain detailed records
- Implement proper risk controls
- Stay updated on regulations
- Start small and scale gradually

## Further Reading

**Books**:
- "Algorithmic and High-Frequency Trading" by Cartea et al.
- "Trading and Exchanges" by Larry Harris
- "The Handbook of Electronic Trading" by Joseph Rosen

**Papers**:
- "High-Frequency Trading and Market Microstructure" (various authors)
- "Optimal Market Making" by Avellaneda & Stoikov
- "Dealing with Inventory Risk" by Guéant et al.

**Resources**:
- Exchange documentation (market making programs)
- SEC/FINRA regulations
- Crypto exchange maker programs

## Warnings

⚠️ **Critical Considerations**:

- Market making is capital intensive
- Requires sophisticated risk management
- Technology failures can be catastrophic
- Regulatory compliance is complex
- Competition from HFT firms
- Start with simulation/paper trading
- Consider costs realistically
- Never trade with money you can't afford to lose

## Next Steps

1. **Paper Trading**
   - Test strategy without risk
   - Understand order flow
   - Refine parameters

2. **Small Scale Live**
   - Start with small size
   - Monitor closely
   - Iterate based on results

3. **Scale Gradually**
   - Increase size slowly
   - Add more instruments
   - Enhance infrastructure

4. **Continuous Improvement**
   - Monitor metrics daily
   - Adjust to market changes
   - Update risk controls
   - Stay educated
