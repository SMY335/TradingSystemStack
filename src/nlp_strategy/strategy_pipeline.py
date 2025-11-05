"""
Complete NLP Strategy Pipeline
NLP → Code → Backtest integration
"""

from .strategy_parser import NLPStrategyParser, StrategyDescription
from .code_generator import StrategyCodeGenerator
from typing import Dict, List, Any
import importlib.util
import sys


class NLPStrategyPipeline:
    """Pipeline complet: NLP → Code → Backtest"""
    
    def __init__(self, api_key: str = None):
        self.parser = NLPStrategyParser(api_key)
        self.generator = StrategyCodeGenerator(api_key)
    
    def create_strategy_from_text(self, description: str) -> Dict[str, Any]:
        """Créer et backtester une stratégie depuis texte naturel"""
        
        print("🤖 Parsing de la stratégie...")
        strategy = self.parser.parse_strategy(description)
        
        print(f"\n📋 Stratégie parsée: {strategy.name}")
        print(f"   Description: {strategy.description}")
        print(f"   Indicateurs: {', '.join(strategy.indicators)}")
        print(f"   Timeframe: {strategy.timeframe}")
        
        # Validation
        print("\n🔍 Validation...")
        issues = self.parser.validate_strategy(strategy)
        if issues:
            print("   Issues détectés:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("   ✅ Aucun problème détecté")
        
        # Génération du code
        print("\n💻 Génération du code...")
        code = self.generator.generate_backtrader_strategy(strategy)
        
        # Sauvegarder
        filename = self.generator.save_strategy(strategy, code)
        print(f"   ✅ Sauvegardé: {filename}")
        
        # Afficher le code
        print("\n📄 Code généré:")
        print("=" * 60)
        print(code)
        print("=" * 60)
        
        return {
            'strategy': strategy,
            'code': code,
            'filename': filename,
            'issues': issues
        }
    
    def backtest_generated_strategy(self, filename: str, 
                                    symbols: List[str] = ['BTC/USDT'],
                                    days: int = 90) -> Dict[str, Any]:
        """Backtester la stratégie générée"""
        
        print(f"\n📊 Backtesting de {filename}...")
        
        # Charger la stratégie dynamiquement
        spec = importlib.util.spec_from_file_location("generated_strategy", filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_strategy"] = module
        spec.loader.exec_module(module)
        
        # Trouver la classe Strategy
        strategy_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name != 'Strategy' and 'Strategy' in str(obj.__bases__):
                strategy_class = obj
                break
        
        if not strategy_class:
            raise ValueError("Aucune classe Strategy trouvée dans le fichier")
        
        print(f"   ✅ Stratégie chargée: {strategy_class.__name__}")
        
        # TODO: Intégrer avec le moteur de backtesting existant
        # Pour l'instant, retourner mock results
        
        return {
            'strategy_class': strategy_class.__name__,
            'final_value': 11234.56,
            'pnl_pct': 12.35,
            'sharpe': 1.45,
            'max_drawdown': -8.2,
            'total_trades': 23,
            'note': 'Backtesting simulation - to be integrated with AdvancedBacktestEngine'
        }
