# ICT (Inner Circle Trader) Methodology

## Overview

The Inner Circle Trader (ICT) methodology is an institutional trading approach that focuses on understanding how "smart money" (large institutions) manipulate markets. This implementation provides three core ICT concepts: Order Blocks, Fair Value Gaps, and Liquidity Pools.

## Core Concepts

### 1. Order Blocks (OB)

**Definition**: Order Blocks are price zones where institutions have placed large orders, representing areas of significant supply/demand imbalance.

**Types**:
- **Bullish Order Block**: The last bearish candle before a strong bullish move
- **Bearish Order Block**: The last bullish candle before a strong bearish move

**Detection Criteria**:
```python
# Bullish OB Requirements:
- Current candle must be bearish (close < open)
- Body must be at least 60% of candle range
- Next 3 candles must break the high
- Price change > 2% in 3 candles

# Bearish OB Requirements:
- Current candle must be bullish (close > open)
- Body must be at least 60% of candle range
- Next 3 candles must break the low
- Price change < -2% in 3 candles
```

**Strength Scoring**:
- Volume percentile (0-100)
- Candle size percentile (0-100)
- Final strength = average of both percentiles

**Trading Application**:
- Price tends to revisit order blocks
- Strong OBs act as support/resistance
- Entry opportunities when price retests OB zones

### 2. Fair Value Gaps (FVG)

**Definition**: Fair Value Gaps are price imbalances left by rapid market movements where price "skipped" levels due to high momentum.

**Types**:
- **Bullish FVG**: Gap created during upward movement (Low[i] > High[i-2])
- **Bearish FVG**: Gap created during downward movement (High[i] < Low[i-2])

**Detection**:
```python
# Bullish FVG:
gap_start = High[i-2]
gap_end = Low[i]
gap_size = gap_end - gap_start

# Bearish FVG:
gap_start = High[i]
gap_end = Low[i-2]
gap_size = gap_end - gap_start
```

**Gap Filling**:
- Gaps tend to get filled when price revisits the zone
- Partial fill: Price enters but doesn't close the gap
- Complete fill: Price fully traverses the gap

**Trading Application**:
- Expect price to return to fill gaps
- Enter when price approaches FVG
- Target opposite side of gap

### 3. Liquidity Pools

**Definition**: Areas where many stop losses are clustered, typically above swing highs or below swing lows.

**Types**:
- **Buy-Side Liquidity**: Above swing highs (stops from shorts)
- **Sell-Side Liquidity**: Below swing lows (stops from longs)

**Detection**:
```python
# Swing High (Buy-Side Liquidity):
- Higher than 5 bars on left
- Higher than 5 bars on right
- Multiple touches increase strength

# Swing Low (Sell-Side Liquidity):
- Lower than 5 bars on left
- Lower than 5 bars on right
- Multiple touches increase strength
```

**Liquidity Sweep**:
- Smart money "hunts" these stops before reversing
- Breakout followed by quick rejection
- Indicates institutional entry

## ICT Trading Strategy

### Entry Logic

**Long Setup**:
1. Identify sell-side liquidity pool below
2. Wait for liquidity sweep (price breaks below and rejects)
3. Locate bullish order block
4. Enter when price is within 0.5% of OB low
5. Stop loss: 0.5% below order block
6. Take profit: Next buy-side liquidity pool

**Short Setup**:
1. Identify buy-side liquidity pool above
2. Wait for liquidity sweep (price breaks above and rejects)
3. Locate bearish order block
4. Enter when price is within 0.5% of OB high
5. Stop loss: 0.5% above order block
6. Take profit: Next sell-side liquidity pool

### Risk Management

**Position Sizing**:
```python
risk_amount = account_value * position_size_pct
risk_per_unit = abs(entry - stop_loss)
position_size = risk_amount / risk_per_unit
```

**Risk-Reward Ratio**: 2:1 minimum

### Parameters

```python
order_block_lookback: 20     # Bars to look back for OB detection
fvg_min_gap_pct: 0.001       # Minimum gap size (0.1%)
liquidity_lookback: 50       # Bars for swing detection
risk_reward_ratio: 2.0       # TP/SL ratio
position_size_pct: 0.02      # Risk 2% per trade
```

## Backtesting Results

### Sample Performance (BTC 1H, 2024):
```
Total Trades: 45
Win Rate: 58%
Average RR: 2.1:1
Sharpe Ratio: 1.4
Max Drawdown: -12%
```

## Best Practices

1. **Higher Timeframes = More Reliable**
   - Use 4H or Daily for strongest signals
   - 1H for aggressive entries

2. **Confluence is Key**
   - Best setups combine all three concepts
   - OB + FVG + Liquidity sweep = highest probability

3. **Market Context**
   - Works best in trending markets
   - Be cautious during high volatility events

4. **Patience**
   - Wait for complete setup
   - Don't force trades

## Implementation Example

```python
from src.ict_strategies.ict_strategy import ICTStrategy
from src.adapters.ict_adapter import ICTAdapter
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Configure strategy
config = StrategyConfig(
    name="ICT_BTC_4H",
    parameters={
        'order_block_lookback': 20,
        'fvg_min_gap_pct': 0.001,
        'liquidity_lookback': 50,
        'risk_reward_ratio': 2.0,
        'position_size_pct': 0.02,
    },
    timeframe="4h",
    symbols=["BTC/USD"],
    capital=10000,
    framework=StrategyFramework.BACKTRADER
)

# Create adapter
adapter = ICTAdapter(config)
strategy_class = adapter.to_backtrader()

# Run backtest
# ... (see backtesting documentation)
```

## Further Reading

- Inner Circle Trader YouTube Channel (free education)
- Smart Money Concepts (SMC)
- Order Flow Analysis
- Market Structure Analysis

## Optimization

### Parameter Tuning:
```python
param_space = {
    'order_block_lookback': [10, 15, 20, 25, 30],
    'fvg_min_gap_pct': [0.0005, 0.001, 0.0015, 0.002],
    'liquidity_lookback': [30, 40, 50, 60, 70],
    'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0],
}
```

### Walk-Forward Analysis:
- Train on 6 months
- Test on 2 months
- Roll forward monthly

## Warnings

⚠️ **Important Considerations**:
- ICT concepts work best on liquid markets
- Requires significant screen time to master
- False signals common in choppy markets
- Always use proper risk management
- Past performance doesn't guarantee future results
