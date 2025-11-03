# Strategy Adapter Layer Documentation

## Overview

The adapter layer provides a unified interface for trading strategies across multiple frameworks (Nautilus Trader and Backtrader). This allows you to write a strategy once and deploy it on any supported framework without code changes.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Strategy Factory                       │
│  (Creates strategies for any framework)                  │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────────┐   ┌────────▼─────────┐
│  EMA Adapter   │   │  RSI Adapter     │
│  RSI Adapter   │   │  MACD Adapter    │
│  MACD Adapter  │   │  ...             │
└───┬────────────┘   └────────┬─────────┘
    │                         │
    │    ┌────────────────────┘
    │    │
┌───▼────▼─────────────────────────────────┐
│      Base Strategy Adapter               │
│  (Abstract interface for all adapters)   │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼──────────┐  ┌───────▼──────────┐
│   Nautilus   │  │   Backtrader     │
│   Strategy   │  │   Strategy       │
└──────────────┘  └──────────────────┘
```

## Why an Adapter Layer?

### Problems Solved

1. **Framework Lock-in**: Without adapters, strategies are tied to a specific framework
2. **Code Duplication**: Same strategy logic needs to be implemented multiple times
3. **Testing Complexity**: Testing across frameworks requires multiple implementations
4. **Migration Difficulty**: Moving from one framework to another requires complete rewrites

### Benefits

1. **Write Once, Deploy Anywhere**: Single strategy implementation works on all frameworks
2. **Consistent Logic**: Same trading rules across backtesting and live trading
3. **Easy Testing**: Test strategy logic once, use on any framework
4. **Framework Flexibility**: Switch frameworks based on needs (backtesting vs live)
5. **Parameter Optimization**: Unified parameter space for optimization

## Components

### 1. Base Strategy Adapter

The foundation of the adapter system. All strategy adapters inherit from this class.

**Key Methods:**
- `validate_parameters()`: Ensures strategy parameters are valid
- `to_nautilus()`: Converts to Nautilus Trader strategy
- `to_backtrader()`: Converts to Backtrader strategy
- `get_parameter_space()`: Returns optimization parameter ranges

**Example:**
```python
from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig

class CustomAdapter(BaseStrategyAdapter):
    def validate_parameters(self):
        # Validate your parameters
        pass
    
    def to_nautilus(self):
        # Return Nautilus strategy
        pass
    
    def to_backtrader(self):
        # Return Backtrader strategy
        pass
    
    def get_parameter_space(self):
        return {'param1': (min, max)}
```

### 2. Strategy Factory

Creates strategy instances for any framework.

**Usage:**
```python
from src.adapters.strategy_factory import StrategyFactory
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Configure strategy
config = StrategyConfig(
    name="my_ema_strategy",
    parameters={'fast_period': 10, 'slow_period': 50},
    timeframe='1h',
    symbols=['BTC/USDT'],
    capital=10000.0,
    framework=StrategyFramework.NAUTILUS
)

# Create strategy
strategy = StrategyFactory.create('ema', config)
```

### 3. Unified Data Manager

Manages data across all frameworks with consistent format.

**Features:**
- Fetches data from CCXT exchanges
- Stores in ArcticDB for fast retrieval
- Converts to Nautilus Bar objects
- Converts to Backtrader data feeds

**Usage:**
```python
from src.infrastructure.data_manager import UnifiedDataManager
from datetime import datetime, timedelta

manager = UnifiedDataManager()

# Fetch and store data
start = datetime.now() - timedelta(days=30)
df = manager.fetch_and_store(
    exchange='binance',
    symbol='BTC/USDT',
    timeframe='1h',
    start_date=start
)

# Retrieve data
df = manager.get_data('BTC/USDT', '1h', start)

# Convert to framework format
nautilus_bars = manager.to_nautilus_bars(df, 'BTC/USDT', '1h')
bt_feed = manager.to_backtrader_feed(df)
```

## Data Flow

```
┌─────────────┐
│   CCXT      │ (Fetch market data)
│  Exchange   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ArcticDB   │ (Store & retrieve)
│   Storage   │
└──────┬──────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│  Nautilus   │    │ Backtrader  │
│    Bars     │    │    Feed     │
└─────────────┘    └─────────────┘
```

## Available Strategies

### 1. EMA Crossover

**Strategy:** Buy when fast EMA crosses above slow EMA, sell when it crosses below.

**Parameters:**
- `fast_period`: Fast EMA period (5-50)
- `slow_period`: Slow EMA period (20-200)

**Config Example:**
```yaml
name: ema_crossover
parameters:
  fast_period: 10
  slow_period: 50
timeframe: 1h
symbols:
  - BTC/USDT
capital: 10000.0
```

### 2. RSI Strategy

**Strategy:** Buy when RSI < oversold, sell when RSI > overbought.

**Parameters:**
- `period`: RSI period (7-28)
- `oversold`: Oversold threshold (20-40)
- `overbought`: Overbought threshold (60-80)

**Config Example:**
```yaml
name: rsi_strategy
parameters:
  period: 14
  oversold: 30
  overbought: 70
timeframe: 1h
symbols:
  - BTC/USDT
capital: 10000.0
```

### 3. MACD Strategy

**Strategy:** Buy when MACD crosses above signal, sell when it crosses below.

**Parameters:**
- `fast_period`: Fast EMA period (8-16)
- `slow_period`: Slow EMA period (20-35)
- `signal_period`: Signal line period (5-15)

**Config Example:**
```yaml
name: macd_strategy
parameters:
  fast_period: 12
  slow_period: 26
  signal_period: 9
timeframe: 1h
symbols:
  - BTC/USDT
capital: 10000.0
```

## Adding a New Strategy

### Step 1: Create Strategy Adapter

```python
# src/adapters/my_strategy_adapter.py
from src.adapters.base_strategy_adapter import BaseStrategyAdapter
from nautilus_trader.trading.strategy import Strategy
import backtrader as bt

class MyNautilusStrategy(Strategy):
    # Implement Nautilus strategy
    pass

class MyBacktraderStrategy(bt.Strategy):
    # Implement Backtrader strategy
    pass

class MyStrategyAdapter(BaseStrategyAdapter):
    def validate_parameters(self):
        # Validate parameters
        pass
    
    def to_nautilus(self):
        return MyNautilusStrategy(self.config)
    
    def to_backtrader(self):
        return MyBacktraderStrategy
    
    def get_parameter_space(self):
        return {'param1': (min, max)}
```

### Step 2: Register with Factory

```python
from src.adapters.strategy_factory import StrategyFactory
from src.adapters.my_strategy_adapter import MyStrategyAdapter

StrategyFactory.register_adapter('my_strategy', MyStrategyAdapter)
```

### Step 3: Create Config File

```yaml
# config/strategies/my_strategy_default.yaml
name: my_strategy
parameters:
  param1: value1
timeframe: 1h
symbols:
  - BTC/USDT
capital: 10000.0
```

## Framework Comparison

### When to Use Nautilus Trader

**Best For:**
- Live trading
- Paper trading
- High-frequency strategies
- Real-time market data
- Production deployment

**Advantages:**
- Built for production
- Real-time event processing
- Advanced order management
- Native exchange connectivity

### When to Use Backtrader

**Best For:**
- Backtesting
- Strategy development
- Historical analysis
- Learning and prototyping

**Advantages:**
- Simple API
- Extensive documentation
- Large community
- Rich indicator library
- Easy to understand

## Usage Examples

### Example 1: Backtest with Backtrader

```python
from src.adapters.strategy_factory import create_ema_strategy
from src.adapters.base_strategy_adapter import StrategyFramework
from src.infrastructure.data_manager import UnifiedDataManager
import backtrader as bt
from datetime import datetime, timedelta

# Get data
manager = UnifiedDataManager()
start = datetime.now() - timedelta(days=30)
df = manager.get_data('BTC/USDT', '1h', start)

# Create strategy
strategy_class = create_ema_strategy(
    fast_period=10,
    slow_period=50,
    framework=StrategyFramework.BACKTRADER
)

# Run backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(strategy_class)
data_feed = manager.to_backtrader_feed(df)
cerebro.adddata(data_feed)
cerebro.broker.setcash(10000.0)
cerebro.run()
```

### Example 2: Live Trading with Nautilus

```python
from src.adapters.strategy_factory import create_ema_strategy
from src.adapters.base_strategy_adapter import StrategyFramework

# Create Nautilus strategy
strategy = create_ema_strategy(
    fast_period=10,
    slow_period=50,
    framework=StrategyFramework.NAUTILUS
)

# Deploy to trading node (see Nautilus documentation)
```

### Example 3: Parameter Optimization

```python
from src.adapters.strategy_factory import StrategyFactory
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Get parameter space
config = StrategyConfig(
    name="ema_opt",
    parameters={'fast_period': 10, 'slow_period': 50},
    timeframe='1h',
    symbols=['BTC/USDT'],
    capital=10000.0,
    framework=StrategyFramework.BACKTRADER
)

adapter = StrategyFactory.get_adapter('ema', config)
param_space = adapter.get_parameter_space()
# {'fast_period': (5, 50), 'slow_period': (20, 200)}

# Use param_space for optimization with your preferred method
```

## Best Practices

1. **Always validate parameters** in your adapter's `validate_parameters()` method
2. **Keep strategy logic identical** across both frameworks
3. **Use native indicators** of each framework (don't reimplement)
4. **Test on both frameworks** to ensure consistency
5. **Store configs in YAML** for easy modification
6. **Use UnifiedDataManager** for all data operations
7. **Document parameter ranges** in `get_parameter_space()`

## Troubleshooting

### Strategy behaves differently on different frameworks

- Check that indicator parameters are identical
- Verify data alignment (timestamps, missing bars)
- Ensure risk parameters are the same
- Compare order execution logic

### Data not found in ArcticDB

- Run migration script: `python scripts/migrate_data_to_arctic.py`
- Check symbol format (use underscores in storage keys)
- Verify date range

### Parameter validation fails

- Check parameter types (int vs float)
- Verify parameter ranges
- Ensure all required parameters are provided

## Future Enhancements

- [ ] Add VectorBT adapter support
- [ ] Support for multi-timeframe strategies
- [ ] Advanced position sizing algorithms
- [ ] Portfolio-level risk management
- [ ] Strategy combinations and ensembles
- [ ] Machine learning strategy adapters

## References

- [Nautilus Trader Documentation](https://nautilustrader.io)
- [Backtrader Documentation](https://www.backtrader.com)
- [ArcticDB Documentation](https://docs.arcticdb.io)
- [CCXT Documentation](https://docs.ccxt.com)
