"""
Strategy Code Generator
Generate Backtrader Python code from StrategyDescription
"""

from .strategy_parser import StrategyDescription
from typing import Dict, Any
import anthropic
import os
import re
import logging

logger = logging.getLogger(__name__)


class StrategyCodeGenerator:
    """Générer le code Python Backtrader depuis StrategyDescription"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate_backtrader_strategy(self, strategy: StrategyDescription) -> str:
        """Générer code Backtrader complet"""
        
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

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        code = message.content[0].text
        
        # Nettoyer le code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def save_strategy(self, strategy: StrategyDescription, code: str, output_dir: str = "src/generated_strategies") -> str:
        """Sauvegarder la stratégie générée"""
        import os

        # SECURITY: Validate strategy name to prevent path traversal attacks
        if not strategy.name or not isinstance(strategy.name, str):
            raise ValueError("Strategy name must be a non-empty string")

        # Only allow alphanumeric characters, spaces, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', strategy.name):
            raise ValueError(
                f"Invalid strategy name: '{strategy.name}'. "
                "Only alphanumeric characters, spaces, hyphens, and underscores are allowed."
            )

        if len(strategy.name) > 100:
            raise ValueError("Strategy name too long (max 100 characters)")

        logger.info(f"Saving strategy: {strategy.name}")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/{strategy.name.lower().replace(' ', '_')}.py"
        
        with open(filename, 'w') as f:
            f.write(f'"""\n')
            f.write(f'{strategy.name}\n')
            f.write(f'\n{strategy.description}\n')
            f.write(f'\nGenerated automatically via NLP Strategy Editor\n')
            f.write(f'"""\n\n')
            f.write(code)
        
        return filename
