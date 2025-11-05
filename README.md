# TradingSystemStack v2.0.0

**Institutional-Grade Algorithmic Trading Framework**

A comprehensive trading system integrating multiple professional frameworks for backtesting, live trading, portfolio optimization, and market making.

## 🎯 Overview

TradingSystemStack combines best-in-class trading frameworks to create a complete institutional trading infrastructure:

- **Nautilus Trader** - High-performance backtesting and live trading
- **Backtrader** - Rapid strategy prototyping
- **Riskfolio-Lib** - Portfolio optimization and risk management
- **ArcticDB** - High-performance time-series database
- **Hummingbot** - Market making and liquidity provision
- **TA-Lib** - Technical analysis library
- **Scikit-learn** - Machine learning capabilities

## ✅ Status

**Installation**: ✅ COMPLETE  
**Validation**: ✅ 100% (9/9 frameworks operational)  
**Documentation**: ✅ COMPLETE  
**Ready for**: Strategy Development

## 🚀 Quick Start

### Run Framework Tests

```bash
python run_framework_tests.py
```

### Initialize ArcticDB

```python
from src.infrastructure.arctic_manager import ArcticManager

manager = ArcticManager()
# 4 libraries ready: market_data, orderbook, trades, backtest_results
```

### Create a Strategy

```python
from nautilus_trader.backtest.engine import BacktestEngine

engine = BacktestEngine()
# Add your strategy here
```

## 📁 Project Structure

```
TradingSystemStack/
├── src/
│   ├── adapters/           # Framework adapters
│   ├── execution/          # Order execution & TCA
│   ├── optimization/       # Strategy optimization
│   ├── ict_strategies/     # ICT trading strategies
│   ├── quant_strategies/   # Quantitative strategies
│   ├── market_making/      # Market making strategies
│   ├── portfolio/          # Portfolio management
│   └── infrastructure/     # Core infrastructure
│
├── data/
│   ├── arctic_db/          # Time-series database
│   ├── raw/                # Raw market data
│   └── processed/          # Processed features
│
├── config/
│   ├── nautilus/           # Nautilus configuration
│   ├── backtrader/         # Backtrader configuration
│   └── strategies/         # Strategy parameters
│
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── performance/        # Performance tests
│
└── docs/
    └── architecture/       # Architecture documentation
```

## 🎯 Key Features

### 📊 Portfolio & Risk Management
- **VaR Analysis**: Historical, Parametric, and Monte Carlo VaR calculations
- **CVaR (Expected Shortfall)**: Advanced tail risk metrics
- **Stress Testing**: Predefined and custom scenario analysis
- **Monte Carlo Simulation**: 10,000+ path simulations
- **Performance Attribution**: Brinson, factor, and risk attribution
- **Interactive Dashboards**: Real-time Streamlit dashboards
- **Telegram Alerts**: Automated risk notifications

### 🔄 Trading Frameworks
- **Multi-Engine Backtesting**: Nautilus Trader, Backtrader, VectorBT
- **Live Trading**: Production-ready execution with Nautilus
- **Market Making**: Hummingbot integration for liquidity provision
- **Strategy Optimization**: Walk-forward analysis and parameter tuning

### 📈 Advanced Strategies
- **ICT Methodology**: Order blocks, liquidity pools, fair value gaps
- **Statistical Arbitrage**: Pairs trading and mean reversion
- **Market Making**: Simple and advanced market making strategies
- **Machine Learning**: Feature engineering and ML integration

## 📊 Installed Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| Nautilus Trader | 1.221.0 | Backtesting & live trading |
| Backtrader | 1.9.78.123 | Strategy prototyping |
| Riskfolio-Lib | 7.0.1 | Portfolio optimization |
| ArcticDB | 6.3.1 | Time-series database |
| Hummingbot | 20250923 | Market making |
| TA-Lib | 0.6.8 | Technical indicators |
| Scikit-learn | 1.7.0 | Machine learning |
| SciPy | 1.16.0 | Scientific computing |
| Statsmodels | 0.14.5 | Statistical analysis |
| Streamlit | 1.29.0+ | Interactive dashboards |

## 📚 Documentation

- **[INSTALLATION_REPORT.md](INSTALLATION_REPORT.md)** - Complete installation summary
- **[docs/architecture/FRAMEWORK_STACK.md](docs/architecture/FRAMEWORK_STACK.md)** - Architecture documentation
- **[pyproject.toml](pyproject.toml)** - Project configuration
- **[requirements_frameworks.txt](requirements_frameworks.txt)** - Installed packages

## 🔧 Configuration

### Nautilus Trader

Configuration file: `config/nautilus/trading_node.yaml`

Key settings:
- Max notional per order: $1,000,000
- Max order rate: 100/second
- Risk checks: ENABLED
- State persistence: ENABLED

### ArcticDB

4 libraries initialized:
- `market_data` - OHLCV data
- `orderbook` - Order book snapshots
- `trades` - Trade records
- `backtest_results` - Performance metrics

## 🧪 Testing

Run all framework tests:

```bash
python run_framework_tests.py
```

Expected output:
```
Passed: 9/9 (100.0%)
🎉 ALL FRAMEWORKS ARE OPERATIONAL!
```

## 🎓 Usage Examples

### Store Market Data

```python
from src.infrastructure.arctic_manager import ArcticManager
import pandas as pd

manager = ArcticManager()

# Write OHLCV data
df = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 
    'close': [...], 'volume': [...]
})
manager.write_market_data("BTC/USDT", df)

# Read market data
data = manager.read_market_data("BTC/USDT")
```

### Run Backtest with Nautilus

```python
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.identifiers import Venue

engine = BacktestEngine()
venue = Venue("BINANCE")
# Add data, strategies, and run backtest
```

### Optimize Portfolio with Riskfolio

```python
import riskfolio as rp
import pandas as pd

# Create portfolio
port = rp.Portfolio(returns=your_returns_df)

# Optimize
weights = port.optimization(model='Classic', rm='MV', obj='Sharpe')
```

### Portfolio Risk Management

```python
from src.portfolio.risk_manager import RiskManager
import pandas as pd

# Load returns data
returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
weights = {'BTC': 0.4, 'ETH': 0.3, 'SOL': 0.2, 'AVAX': 0.1}

# Initialize risk manager
risk_manager = RiskManager(returns, weights)

# Calculate comprehensive risk metrics
metrics = risk_manager.calculate_risk_metrics()
print(f"VaR 95%: {metrics.var_95:.2%}")
print(f"CVaR 95%: {metrics.cvar_95:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

# Run stress tests
stress_results = risk_manager.stress_test()

# Monte Carlo simulation
final_values, paths = risk_manager.monte_carlo_simulation(n_simulations=10000)

# Generate report
report = risk_manager.generate_risk_report()
print(report)
```

### Performance Attribution

```python
from src.portfolio.performance_attribution import PerformanceAttributor

# Initialize attributor
attributor = PerformanceAttributor(returns, portfolio_weights, benchmark_weights)

# Brinson attribution
attribution = attributor.brinson_attribution()
print(f"Active Return: {attribution.active_return:.2%}")
print(f"Allocation Effect: {attribution.allocation_effect:.2%}")
print(f"Selection Effect: {attribution.selection_effect:.2%}")

# Factor attribution
factor_contrib = attributor.factor_attribution()

# Risk attribution
risk_attr = attributor.risk_attribution()
print(f"Portfolio Risk: {risk_attr.portfolio_risk:.2%}")
print(f"Diversification Ratio: {risk_attr.diversification_ratio:.2f}")

# Generate report
report = attributor.generate_attribution_report()
```

### Launch Interactive Dashboards

```bash
# Risk Management Dashboard
./scripts/launch_risk_dashboard.sh

# Performance Attribution Dashboard
./scripts/launch_attribution_dashboard.sh

# Or run complete analysis
python scripts/run_portfolio_analysis.py --output reports/ --telegram
```

## 🛠️ Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Format Code

```bash
black src/ tests/
```

### Lint Code

```bash
ruff check src/ tests/
```

## 📈 Performance

- **Nautilus Trader**: 77ms initialization, Rust core for critical paths
- **ArcticDB**: 10-100x faster than SQL for time-series queries
- **Memory**: Efficient with chunked data loading
- **Scalability**: Horizontal scaling ready

## 🚦 Next Steps

1. **Strategy Development** - Implement your first trading strategy
2. **Data Pipeline** - Set up market data ingestion from exchanges
3. **Backtesting** - Run historical simulations
4. **Portfolio Optimization** - Optimize strategy allocation
5. **Risk Management** - Configure position limits and alerts
6. **Live Trading** - Deploy to production (paper trading first!)

## 📞 Support

- Check [INSTALLATION_REPORT.md](INSTALLATION_REPORT.md) for detailed setup info
- Review [docs/architecture/FRAMEWORK_STACK.md](docs/architecture/FRAMEWORK_STACK.md) for architecture
- Run `python run_framework_tests.py` to validate installation

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Use at your own risk.

---

**Version**: 2.0.0  
**Status**: Production Ready  
**Python**: 3.11+  
**Last Updated**: November 2025
