"""
NLP Strategy Parser using Claude API
Parse natural language descriptions into structured strategy definitions
"""

import anthropic
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import os


@dataclass
class StrategyDescription:
    """Structured representation of a trading strategy"""
    name: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    indicators: List[str]
    timeframe: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class NLPStrategyParser:
    """
    Parse stratégies en langage naturel via Claude API
    
    Exemple input:
    "Je veux une stratégie qui achète quand le RSI est en dessous de 30 
     et vend quand il dépasse 70. Utilise un stop loss de 2% et 
     take profit de 5%. Trade sur 1h."
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY manquante")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def parse_strategy(self, natural_language_description: str) -> StrategyDescription:
        """Convertir description en langage naturel vers structure"""
        
        prompt = f"""Tu es un expert en trading algorithmique. 

Analyse cette description de stratégie de trading et extrais les informations structurées:

DESCRIPTION: {natural_language_description}

Retourne un JSON avec cette structure exacte:
{{
    "name": "nom court de la stratégie",
    "entry_conditions": ["condition 1", "condition 2", ...],
    "exit_conditions": ["condition 1", "condition 2", ...],
    "risk_management": {{
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.05,
        "position_size_pct": 0.1,
        "max_positions": 1
    }},
    "indicators": ["RSI", "EMA", "MACD", etc.],
    "timeframe": "1h",
    "description": "description claire de la logique"
}}

Règles:
- entry_conditions et exit_conditions doivent être des conditions booléennes claires
- Utilise les indicateurs standards: RSI, EMA, SMA, MACD, Bollinger Bands, ATR, Volume
- stop_loss_pct et take_profit_pct en décimal (2% = 0.02)
- timeframe: 1m, 5m, 15m, 1h, 4h, 1d
- Si une info n'est pas spécifiée, utilise des valeurs par défaut raisonnables

Retourne UNIQUEMENT le JSON, sans texte autour."""

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parser la réponse
        response_text = message.content[0].text
        
        # Nettoyer le JSON (enlever markdown si présent)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        strategy_dict = json.loads(response_text.strip())
        
        return StrategyDescription(
            name=strategy_dict['name'],
            entry_conditions=strategy_dict['entry_conditions'],
            exit_conditions=strategy_dict['exit_conditions'],
            risk_management=strategy_dict['risk_management'],
            indicators=strategy_dict['indicators'],
            timeframe=strategy_dict['timeframe'],
            description=strategy_dict['description']
        )
    
    def validate_strategy(self, strategy: StrategyDescription) -> List[str]:
        """Valider la stratégie et retourner warnings/erreurs"""
        issues = []
        
        # Vérifier conditions
        if not strategy.entry_conditions:
            issues.append("❌ Aucune condition d'entrée définie")
        
        if not strategy.exit_conditions:
            issues.append("⚠️  Aucune condition de sortie définie")
        
        # Vérifier risk management
        rm = strategy.risk_management
        if rm.get('stop_loss_pct', 0) > 0.1:
            issues.append("⚠️  Stop loss > 10% est très large")
        
        if rm.get('take_profit_pct', 0) < rm.get('stop_loss_pct', 0):
            issues.append("⚠️  Take profit < Stop loss (ratio risque/rendement < 1)")
        
        # Vérifier indicateurs
        supported_indicators = ['RSI', 'EMA', 'SMA', 'MACD', 'BB', 'ATR', 'VOLUME', 'VWAP', 'STOCH']
        for indicator in strategy.indicators:
            if indicator.upper() not in supported_indicators:
                issues.append(f"⚠️  Indicateur '{indicator}' non supporté")
        
        return issues
