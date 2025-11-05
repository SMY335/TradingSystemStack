"""
Strategy Code Generator
Generate Backtrader Python code from StrategyDescription
"""

from .strategy_parser import StrategyDescription
from typing import Dict, Any
import anthropic
import logging
import re
import os

# Configure logging
logger = logging.getLogger(__name__)

# Valid timeframes
VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M']

# Valid indicators
VALID_INDICATORS = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'ADX', 'Stochastic', 'CCI', 'MOM', 'ROC']


class StrategyCodeGenerator:
    """Générer le code Python Backtrader depuis StrategyDescription"""
    
    def __init__(self, api_key: str = None):
        # Validate api_key
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment variables")
        
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("StrategyCodeGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise ConnectionError(f"Failed to initialize Anthropic client") from e
    
    def generate_backtrader_strategy(self, strategy: StrategyDescription) -> str:
        """Générer code Backtrader complet"""
        
        # Validate strategy object
        if strategy is None:
            raise ValueError("strategy cannot be None")
        
        if not isinstance(strategy, StrategyDescription):
            raise TypeError(f"strategy must be StrategyDescription, got {type(strategy)}")
        
        # Validate name
        if not strategy.name or not isinstance(strategy.name, str):
            raise ValueError("strategy.name must be a non-empty string")
        
        # Sanitize name (alphanumeric + underscore only)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', strategy.name)
        if safe_name != strategy.name:
            logger.warning(f"Strategy name sanitized: '{strategy.name}' -> '{safe_name}'")
            strategy.name = safe_name
        
        # Ensure name starts with letter
        if not safe_name[0].isalpha():
            strategy.name = 'Strategy_' + safe_name
            logger.warning(f"Strategy name adjusted to start with letter: '{strategy.name}'")
        
        # Validate name length
        if len(strategy.name) > 100:
            raise ValueError(f"strategy.name too long: {len(strategy.name)} chars. Maximum: 100")
        
        # Validate description
        if not strategy.description or not isinstance(strategy.description, str):
            raise ValueError("strategy.description must be a non-empty string")
        
        if len(strategy.description) > 1000:
            raise ValueError(f"strategy.description too long: {len(strategy.description)} chars. Maximum: 1000")
        
        # Validate entry_conditions
        if not strategy.entry_conditions or not isinstance(strategy.entry_conditions, list):
            raise ValueError("strategy.entry_conditions must be a non-empty list")
        
        if len(strategy.entry_conditions) == 0:
            raise ValueError("strategy must have at least one entry_condition")
        
        if len(strategy.entry_conditions) > 10:
            raise ValueError(f"Too many entry_conditions: {len(strategy.entry_conditions)}. Maximum: 10")
        
        for i, cond in enumerate(strategy.entry_conditions):
            if not isinstance(cond, str) or not cond.strip():
                raise ValueError(f"entry_condition {i} must be a non-empty string")
            
            if len(cond) > 500:
                raise ValueError(f"entry_condition {i} too long: {len(cond)} chars. Maximum: 500")
        
        # Validate exit_conditions
        if not strategy.exit_conditions or not isinstance(strategy.exit_conditions, list):
            raise ValueError("strategy.exit_conditions must be a non-empty list")
        
        if len(strategy.exit_conditions) == 0:
            raise ValueError("strategy must have at least one exit_condition")
        
        if len(strategy.exit_conditions) > 10:
            raise ValueError(f"Too many exit_conditions: {len(strategy.exit_conditions)}. Maximum: 10")
        
        for i, cond in enumerate(strategy.exit_conditions):
            if not isinstance(cond, str) or not cond.strip():
                raise ValueError(f"exit_condition {i} must be a non-empty string")
            
            if len(cond) > 500:
                raise ValueError(f"exit_condition {i} too long: {len(cond)} chars. Maximum: 500")
        
        # Validate timeframe
        if not strategy.timeframe or not isinstance(strategy.timeframe, str):
            raise ValueError("strategy.timeframe must be a non-empty string")
        
        if strategy.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{strategy.timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        # Validate indicators
        if strategy.indicators:
            if not isinstance(strategy.indicators, list):
                raise TypeError(f"strategy.indicators must be list, got {type(strategy.indicators)}")
            
            if len(strategy.indicators) > 10:
                raise ValueError(f"Too many indicators: {len(strategy.indicators)}. Maximum: 10")
            
            invalid_indicators = [ind for ind in strategy.indicators if ind not in VALID_INDICATORS]
            if invalid_indicators:
                raise ValueError(
                    f"Invalid indicators: {invalid_indicators}. Valid indicators: {VALID_INDICATORS}"
                )
        
        # Validate risk_management
        if strategy.risk_management:
            if not isinstance(strategy.risk_management, dict):
                raise TypeError(f"risk_management must be dict, got {type(strategy.risk_management)}")
            
            # Validate stop_loss_pct
            if 'stop_loss_pct' in strategy.risk_management:
                sl = strategy.risk_management['stop_loss_pct']
                if not isinstance(sl, (int, float)):
                    raise TypeError(f"stop_loss_pct must be numeric, got {type(sl)}")
                
                if sl < 0 or sl > 0.5:
                    raise ValueError(f"stop_loss_pct must be between 0 and 0.5 (0-50%), got {sl}")
            
            # Validate take_profit_pct
            if 'take_profit_pct' in strategy.risk_management:
                tp = strategy.risk_management['take_profit_pct']
                if not isinstance(tp, (int, float)):
                    raise TypeError(f"take_profit_pct must be numeric, got {type(tp)}")
                
                if tp < 0 or tp > 2.0:
                    raise ValueError(f"take_profit_pct must be between 0 and 2.0 (0-200%), got {tp}")
            
            # Validate position_size_pct
            if 'position_size_pct' in strategy.risk_management:
                ps = strategy.risk_management['position_size_pct']
                if not isinstance(ps, (int, float)):
                    raise TypeError(f"position_size_pct must be numeric, got {type(ps)}")
                
                if ps <= 0 or ps > 1.0:
                    raise ValueError(f"position_size_pct must be between 0 and 1.0 (0-100%), got {ps}")
        
        logger.info(f"Generating Backtrader strategy code for: {strategy.name}")
        
        prompt = f"""Tu es un expert en Backtrader (framework Python de backtesting).

Génère une stratégie Backtrader COMPLÈTE et FONCTIONNELLE basée sur cette description:

NOM: {strategy.name}
DESCRIPTION: {strategy.description}

CONDITIONS D'ENTRÉE:
{chr(10).join('- ' + cond for cond in strategy.entry_conditions)}

CONDITIONS DE SORTIE:
{chr(10).join('- ' + cond for cond in strategy.exit_conditions)}

INDICATEURS: {', '.join(strategy.indicators)}

RISK MANAGEMENT:
- Stop Loss: {strategy.risk_management.get('stop_loss_pct', 0.02)*100}%
- Take Profit: {strategy.risk_management.get('take_profit_pct', 0.05)*100}%
- Position Size: {strategy.risk_management.get('position_size_pct', 0.1)*100}%

TIMEFRAME: {strategy.timeframe}

Génère le code Python avec:
1. Import backtrader as bt
2. Classe héritant de bt.Strategy
3. Initialisation des indicateurs dans __init__
4. Logique de trading dans next()
5. Gestion des ordres dans notify_order()
6. Stop loss et take profit implémentés
7. Position sizing basé sur account value
8. Logs clairs des trades

RETOURNE UNIQUEMENT LE CODE PYTHON, sans explications."""

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            code = message.content[0].text
        except Exception as e:
            logger.error(f"Failed to generate code from API: {e}")
            raise RuntimeError(f"Failed to generate strategy code") from e
        
        # Nettoyer le code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        code = code.strip()
        
        # Validate generated code is not empty
        if not code:
            raise ValueError("Generated code is empty")
        
        # Basic validation that code looks like Python
        if "import" not in code or "class" not in code:
            logger.warning("Generated code may not be valid Python strategy")
        
        logger.info(f"Successfully generated {len(code)} characters of code for {strategy.name}")
        
        return code
    
    def save_strategy(self, strategy: StrategyDescription, code: str, output_dir: str = "src/generated_strategies") -> str:
        """Sauvegarder la stratégie générée"""
        # Validate strategy
        if strategy is None:
            raise ValueError("strategy cannot be None")
        
        if not isinstance(strategy, StrategyDescription):
            raise TypeError(f"strategy must be StrategyDescription, got {type(strategy)}")
        
        # Validate code
        if not code or not isinstance(code, str):
            raise ValueError("code must be a non-empty string")
        
        if len(code) < 50:
            raise ValueError(f"code too short: {len(code)} chars. Minimum: 50")
        
        # Validate output_dir
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir must be a non-empty string")
        
        # Sanitize output_dir path
        output_dir = output_dir.strip()
        if '..' in output_dir or output_dir.startswith('/'):
            raise ValueError(f"Invalid output_dir path: {output_dir}")
        
        # Create directory safely
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {output_dir}: {e}")
            raise IOError(f"Failed to create output directory") from e
        
        # Sanitize filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_]', '_', strategy.name.lower())
        filename = f"{output_dir}/{safe_filename}.py"
        
        # Prevent path traversal
        abs_output_dir = os.path.abspath(output_dir)
        abs_filename = os.path.abspath(filename)
        if not abs_filename.startswith(abs_output_dir):
            raise ValueError(f"Security violation: filename outside output_dir")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'{strategy.name}\n')
                f.write(f'\n{strategy.description}\n')
                f.write(f'\nGenerated automatically via NLP Strategy Editor\n')
                f.write(f'"""\n\n')
                f.write(code)
            
            logger.info(f"Strategy saved successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save strategy to {filename}: {e}")
            raise IOError(f"Failed to save strategy file") from e
