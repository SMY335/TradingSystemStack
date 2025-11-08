# Scanner DSL Reference

The TradingSystemStack scanner provides a JSON-based Domain Specific Language (DSL) for defining market scans declaratively.

## Overview

Scans allow you to filter and rank securities based on technical indicators, price action, and patterns. The scanner executes scans in parallel for optimal performance.

## Quick Start

### Example: RSI Oversold Scanner

```json
{
  "name": "RSI Oversold Scanner",
  "universe": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "timeframe": "1d",
  "lookback": 100,
  "indicators": [
    {
      "name": "RSI",
      "params": {"period": 14},
      "alias": "rsi"
    }
  ],
  "conditions": {
    "type": "comparison",
    "left": "rsi",
    "operator": "<",
    "right": 30
  },
  "sort_by": "rsi",
  "sort_ascending": true,
  "max_results": 10
}
```

### Running a Scan

```python
from src.scanner import load_scan_from_json, run_scan

# Load and execute
scan = load_scan_from_json('scans/rsi_oversold.json')
results = run_scan(scan, verbose=True)

# Access results
for match in results.matched:
    print(f"{match.symbol}: RSI={match.values['rsi']:.2f}")
```

## Scan Definition Schema

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Scan name |
| `description` | string | | Scan description |
| `universe` | array[string] | ✓ | List of symbols to scan |
| `timeframe` | string | | Timeframe (1m, 5m, 1h, 1d, 1w). Default: "1d" |
| `lookback` | integer | | Number of bars to load. Default: 100 |
| `indicators` | array[Indicator] | | Indicators to calculate |
| `conditions` | Condition | ✓ | Condition tree to match |
| `max_results` | integer | | Maximum results to return |
| `sort_by` | string | | Column to sort results by |
| `sort_ascending` | boolean | | Sort direction. Default: false |

### Indicator Configuration

```json
{
  "name": "SMA",
  "params": {
    "period": 50
  },
  "alias": "sma_50"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Indicator name (RSI, SMA, EMA, MACD, etc.) |
| `params` | object | | Indicator parameters |
| `alias` | string | | Alias for referencing in conditions |

**Supported Indicators:**
- SMA, EMA, WMA - Moving averages
- RSI - Relative Strength Index
- MACD - Moving Average Convergence Divergence
- ATR - Average True Range
- BBANDS - Bollinger Bands
- STOCH - Stochastic Oscillator
- All indicators from `src.indicators` module

## Condition Types

### 1. Comparison

Compare a value against a constant or another column.

```json
{
  "type": "comparison",
  "left": "rsi",
  "operator": "<",
  "right": 30
}
```

**Operators:** `>`, `<`, `>=`, `<=`, `==`, `!=`

**Examples:**

```json
// Price above SMA
{
  "type": "comparison",
  "left": "close",
  "operator": ">",
  "right": "sma_50"
}

// Volume above 1 million
{
  "type": "comparison",
  "left": "volume",
  "operator": ">",
  "right": 1000000
}
```

### 2. Crossover/Crossunder

Detect when one series crosses above or below another.

```json
{
  "type": "cross",
  "series1": "sma_50",
  "series2": "sma_200",
  "direction": "above",
  "lookback": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | "cross" | ✓ | Condition type |
| `series1` | string | ✓ | Fast series (crosses over) |
| `series2` | string | ✓ | Slow series (crossed) |
| `direction` | "above" or "below" | ✓ | Cross direction |
| `lookback` | integer | | Bars to look back. Default: 5 |

### 3. Pattern Detection

Detect candlestick or chart patterns.

```json
{
  "type": "pattern",
  "pattern_type": "doji",
  "timeframe": "1d"
}
```

**Supported Patterns:**
- doji, hammer, shooting_star, engulfing
- morning_star, evening_star
- three_white_soldiers, three_black_crows

### 4. Logical Operators

Combine multiple conditions with AND/OR logic.

#### AND Condition

All sub-conditions must be true.

```json
{
  "type": "and",
  "conditions": [
    {
      "type": "comparison",
      "left": "close",
      "operator": ">",
      "right": "sma_50"
    },
    {
      "type": "comparison",
      "left": "rsi",
      "operator": ">",
      "right": 60
    }
  ]
}
```

#### OR Condition

At least one sub-condition must be true.

```json
{
  "type": "or",
  "conditions": [
    {
      "type": "comparison",
      "left": "rsi",
      "operator": "<",
      "right": 30
    },
    {
      "type": "comparison",
      "left": "rsi",
      "operator": ">",
      "right": 70
    }
  ]
}
```

## Complete Examples

### Golden Cross Scanner

```json
{
  "name": "Golden Cross Scanner",
  "description": "Find stocks where SMA(50) crosses above SMA(200)",
  "universe": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL"],
  "timeframe": "1d",
  "lookback": 250,
  "indicators": [
    {
      "name": "SMA",
      "params": {"period": 50},
      "alias": "sma_50"
    },
    {
      "name": "SMA",
      "params": {"period": 200},
      "alias": "sma_200"
    }
  ],
  "conditions": {
    "type": "cross",
    "series1": "sma_50",
    "series2": "sma_200",
    "direction": "above",
    "lookback": 5
  },
  "max_results": 20
}
```

### Momentum Breakout Scanner

```json
{
  "name": "Momentum Breakout",
  "description": "Strong momentum with breakout confirmation",
  "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
  "timeframe": "1d",
  "lookback": 100,
  "indicators": [
    {
      "name": "SMA",
      "params": {"period": 50},
      "alias": "sma_50"
    },
    {
      "name": "RSI",
      "params": {"period": 14},
      "alias": "rsi"
    },
    {
      "name": "SMA",
      "params": {"period": 20, "field": "volume"},
      "alias": "avg_volume"
    }
  ],
  "conditions": {
    "type": "and",
    "conditions": [
      {
        "type": "comparison",
        "left": "close",
        "operator": ">",
        "right": "sma_50"
      },
      {
        "type": "comparison",
        "left": "rsi",
        "operator": ">",
        "right": 60
      },
      {
        "type": "comparison",
        "left": "volume",
        "operator": ">",
        "right": "avg_volume"
      }
    ]
  },
  "sort_by": "rsi",
  "sort_ascending": false,
  "max_results": 15
}
```

## Python API

### Creating Scans Programmatically

```python
from src.scanner import create_simple_scan, create_cross_scan

# Simple single-indicator scan
scan = create_simple_scan(
    name='RSI Oversold',
    universe=['AAPL', 'MSFT', 'GOOGL'],
    indicator_name='RSI',
    operator='<',
    threshold=30
)

# Crossover scan
scan = create_cross_scan(
    name='Golden Cross',
    universe=['SPY', 'QQQ'],
    fast_indicator='sma_50',
    slow_indicator='sma_200',
    direction='above'
)
```

### Advanced Scan Creation

```python
from src.scanner import ScanDefinition, IndicatorConfig, LogicalCondition, Comparison

scan = ScanDefinition(
    name='Custom Multi-Condition Scan',
    universe=['AAPL', 'MSFT', 'GOOGL'],
    timeframe='1d',
    lookback=100,
    indicators=[
        IndicatorConfig(name='RSI', params={'period': 14}, alias='rsi'),
        IndicatorConfig(name='SMA', params={'period': 50}, alias='sma')
    ],
    conditions=LogicalCondition(
        type='and',
        conditions=[
            Comparison(type='comparison', left='rsi', operator='>', right=50),
            Comparison(type='comparison', left='close', operator='>', right='sma')
        ]
    ),
    sort_by='rsi',
    max_results=10
)
```

### Running Scans

```python
from src.scanner import ScanEngine, run_scan

# Using convenience function
results = run_scan(scan, max_workers=4, verbose=True)

# Using engine directly
engine = ScanEngine(max_workers=4, verbose=True)
results = engine.execute(scan)

# Access results
print(f"Found {len(results.matched)} matches")
print(f"Execution time: {results.execution_time:.2f}s")

for result in results.matched:
    print(f"{result.symbol}: {result.values}")
```

### Saving and Loading Scans

```python
from src.scanner import save_scan_to_json, load_scan_from_json

# Save to JSON
save_scan_to_json(scan, 'scans/my_scan.json')

# Load from JSON
scan = load_scan_from_json('scans/my_scan.json')
```

## Performance Tips

1. **Limit Universe Size**: Scan fewer symbols for faster results
2. **Reduce Lookback**: Use the minimum lookback period needed
3. **Parallel Execution**: Set `max_workers=4` or higher for CPU-bound scans
4. **Filter Early**: Put most restrictive conditions first in AND logic
5. **Cache Results**: Save scan results to avoid re-running

## Troubleshooting

### Scan Returns No Results

- Check that universe symbols are valid
- Verify conditions are not too restrictive
- Ensure indicators are calculating correctly
- Check that timeframe data is available

### Slow Performance

- Reduce universe size
- Lower lookback period
- Increase `max_workers` for parallel execution
- Simplify indicator calculations

### Missing Indicator Values

- Ensure indicator alias matches condition references
- Check that indicators have enough data (lookback period)
- Verify indicator parameters are correct

## See Also

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Indicators Module](../src/indicators/) - Available indicators
- [Examples](../examples/scans/) - More scan examples
