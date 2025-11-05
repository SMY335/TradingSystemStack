# NLP Strategy Editor

## Overview

The NLP Strategy Editor allows traders to create trading strategies using natural language descriptions, powered by Claude AI. The system automatically parses descriptions, generates Backtrader-compatible Python code, validates strategies, and prepares them for backtesting.

## Features

- **Natural Language Parsing**: Describe strategies in plain English or French
- **Automatic Code Generation**: Generates complete Backtrader strategy code
- **Strategy Validation**: Identifies potential issues and warnings
- **Multiple Indicators**: Supports RSI, MACD, EMA, SMA, Bollinger Bands, ATR, and more
- **Risk Management**: Automatic stop-loss and take-profit implementation
- **Interactive Dashboard**: Streamlit-based UI for easy strategy creation

## Architecture

```
User Input (Natural Language)
    ↓
NLPStrategyParser (Claude AI)
    ↓
StrategyDescription (Structured Data)
    ↓
StrategyCodeGenerator (Claude AI)
    ↓
Python Strategy Code (Backtrader)
    ↓
Validation & Backtesting
```

## Usage

### 1. Command Line

```python
from src.nlp_strategy.strategy_pipeline import NLPStrategyPipeline

# Initialize pipeline with API key
pipeline = NLPStrategyPipeline(api_key="your-anthropic-api-key")

# Create strategy from text
description = """
I want a strategy that buys when RSI drops below 30 
and sells when it exceeds 70. Use 2% stop loss and 
4% take profit. Trade on 1h timeframe.
"""

result = pipeline.create_strategy_from_text(description)

# Access generated code
print(result['code'])
print(f"Strategy saved to: {result['filename']}")
```

### 2. Streamlit Dashboard

```bash
streamlit run src/dashboard/nlp_strategy_editor.py
```

Then:
1. Enter your Anthropic API key in the sidebar
2. Describe your strategy in natural language
3. Click "Generate Strategy"
4. Review the generated code and details
5. Download or save the strategy

## Example Descriptions

### RSI Oversold/Overbought

```
Je veux une stratégie qui achète quand le RSI tombe en dessous de 30 
et vend quand il dépasse 70. Utilise un stop loss de 2% et take profit de 4%. 
Trade sur timeframe 1h.
```

### EMA Crossover

```
Create a strategy that buys when EMA 10 crosses above EMA 50 
AND volume is above average. Exit when EMA 10 crosses below EMA 50. 
Stop loss 3%, take profit 6%. Timeframe 4h.
```

### Bollinger Bands

```
I want to buy when price touches the lower Bollinger Band 
and RSI < 40. Sell when price reaches the middle band. 
Stop loss 1.5%, take profit 3%. Timeframe 1h.
```

### MACD Momentum

```
Strategy that enters long when MACD line crosses above signal 
AND price is above EMA 200. Exit when MACD crosses below signal. 
Stop loss 2.5%, take profit 5%. Timeframe 1h.
```

## Strategy Structure

Generated strategies follow this structure:

```python
{
    "name": "Strategy Name",
    "description": "Clear description of logic",
    "entry_conditions": [
        "Condition 1",
        "Condition 2"
    ],
    "exit_conditions": [
        "Condition 1",
        "Condition 2"
    ],
    "risk_management": {
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
        "position_size_pct": 0.1,
        "max_positions": 1
    },
    "indicators": ["RSI", "EMA", "MACD"],
    "timeframe": "1h"
}
```

## Supported Indicators

- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: Volume, OBV, VWAP

## Supported Timeframes

- 1m (1 minute)
- 5m (5 minutes)
- 15m (15 minutes)
- 1h (1 hour)
- 4h (4 hours)
- 1d (1 day)

## Validation Rules

The system automatically validates:

1. **Entry Conditions**: Must have at least one entry condition
2. **Exit Conditions**: Warning if no exit conditions
3. **Stop Loss**: Warning if > 10%
4. **Risk/Reward**: Warning if take profit < stop loss
5. **Indicators**: Check if indicators are supported

## API Reference

### NLPStrategyParser

```python
class NLPStrategyParser:
    def __init__(self, api_key: str = None)
    def parse_strategy(self, description: str) -> StrategyDescription
    def validate_strategy(self, strategy: StrategyDescription) -> List[str]
```

### StrategyCodeGenerator

```python
class StrategyCodeGenerator:
    def __init__(self, api_key: str = None)
    def generate_backtrader_strategy(self, strategy: StrategyDescription) -> str
    def save_strategy(self, strategy: StrategyDescription, code: str, 
                     output_dir: str = "src/generated_strategies") -> str
```

### NLPStrategyPipeline

```python
class NLPStrategyPipeline:
    def __init__(self, api_key: str = None)
    def create_strategy_from_text(self, description: str) -> Dict[str, Any]
    def backtest_generated_strategy(self, filename: str, symbols: List[str], 
                                    days: int = 90) -> Dict[str, Any]
```

## Configuration

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your-key-here"
```

### Output Directory

Generated strategies are saved to:
```
src/generated_strategies/
```

## Best Practices

1. **Be Specific**: Include exact threshold values (e.g., "RSI < 30" not "RSI is low")
2. **Define Timeframes**: Always specify the timeframe
3. **Include Risk Management**: Specify stop loss and take profit
4. **Test Thoroughly**: Always backtest generated strategies before live trading
5. **Review Code**: Verify the generated code matches your intent

## Limitations

- Requires valid Anthropic API key
- Limited to supported indicators
- Generated code may need manual refinement
- Backtesting integration is in progress

## Troubleshooting

### API Key Issues

```python
# Check if API key is set
import os
print(os.environ.get('ANTHROPIC_API_KEY'))
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install anthropic streamlit
```

### Generation Failures

- Check API key validity
- Verify description clarity
- Review Claude API status

## Future Enhancements

- [ ] Real-time backtesting integration
- [ ] Multi-timeframe strategies
- [ ] Advanced order types
- [ ] Portfolio strategies
- [ ] ML-enhanced signal generation
- [ ] Strategy optimization suggestions

## Support

For issues or questions:
- Check the documentation
- Review example strategies
- Test with provided examples
- Verify API key configuration
