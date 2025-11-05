"""
Moteur de backtesting avancé avec Backtrader pour niveau institutionnel
"""
import backtrader as bt
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from src.infrastructure.data_manager import UnifiedDataManager
from src.adapters.strategy_factory import StrategyFactory
from src.adapters.base_strategy_adapter import StrategyConfig, StrategyFramework

# Configure logging
logger = logging.getLogger(__name__)

# Valid timeframes for backtesting
VALID_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']


@dataclass
class BacktestConfig:
    """Configuration du backtest"""
    strategy_name: str
    strategy_params: Dict[str, Any]
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage_percent: float = 0.0005  # 0.05%
    position_size_percent: float = 0.95
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Validate strategy_name
        if not self.strategy_name or not isinstance(self.strategy_name, str):
            raise ValueError("strategy_name must be a non-empty string")
        
        # Validate strategy_params
        if not isinstance(self.strategy_params, dict):
            raise TypeError(f"strategy_params must be dict, got {type(self.strategy_params)}")
        
        # Validate symbols
        if not self.symbols or not isinstance(self.symbols, list):
            raise ValueError("symbols must be a non-empty list")
        
        if not all(isinstance(s, str) and s for s in self.symbols):
            raise ValueError("All symbols must be non-empty strings")
        
        # Validate timeframe
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Invalid timeframe '{self.timeframe}'. Must be one of {VALID_TIMEFRAMES}"
            )
        
        # Validate dates
        if not isinstance(self.start_date, datetime):
            raise TypeError(f"start_date must be datetime, got {type(self.start_date)}")
        
        if not isinstance(self.end_date, datetime):
            raise TypeError(f"end_date must be datetime, got {type(self.end_date)}")
        
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )
        
        if self.end_date > datetime.now():
            raise ValueError(f"end_date ({self.end_date}) cannot be in the future")
        
        # Validate initial_capital
        if not isinstance(self.initial_capital, (int, float)):
            raise TypeError(f"initial_capital must be numeric, got {type(self.initial_capital)}")
        
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if self.initial_capital < 100:
            logger.warning(f"initial_capital is very low: ${self.initial_capital}")
        
        # Validate commission
        if not isinstance(self.commission, (int, float)):
            raise TypeError(f"commission must be numeric, got {type(self.commission)}")
        
        if self.commission < 0 or self.commission > 1:
            raise ValueError(f"commission must be between 0 and 1 (0-100%), got {self.commission}")
        
        # Validate slippage_percent
        if not isinstance(self.slippage_percent, (int, float)):
            raise TypeError(f"slippage_percent must be numeric, got {type(self.slippage_percent)}")
        
        if self.slippage_percent < 0 or self.slippage_percent > 0.1:
            raise ValueError(f"slippage_percent must be between 0 and 0.1 (0-10%), got {self.slippage_percent}")
        
        # Validate position_size_percent
        if not isinstance(self.position_size_percent, (int, float)):
            raise TypeError(f"position_size_percent must be numeric, got {type(self.position_size_percent)}")
        
        if self.position_size_percent <= 0 or self.position_size_percent > 1:
            raise ValueError(f"position_size_percent must be between 0 and 1, got {self.position_size_percent}")
        
        logger.info(f"BacktestConfig validated: {self.strategy_name} on {len(self.symbols)} symbols")
    

@dataclass
class TransactionCosts:
    """Modèle de coûts de transaction réaliste (Binance)"""
    maker_fee: float = 0.0002  # 0.02% Binance maker
    taker_fee: float = 0.0004  # 0.04% Binance taker
    slippage_model: str = "fixed"  # "fixed", "volumetric", "sqrt"
    slippage_basis_points: float = 5  # 0.05%
    
    def __post_init__(self):
        """Validate transaction cost parameters"""
        # Validate maker_fee
        if not isinstance(self.maker_fee, (int, float)):
            raise TypeError(f"maker_fee must be numeric, got {type(self.maker_fee)}")
        
        if self.maker_fee < 0 or self.maker_fee > 0.01:
            raise ValueError(f"maker_fee must be between 0 and 0.01 (0-1%), got {self.maker_fee}")
        
        # Validate taker_fee
        if not isinstance(self.taker_fee, (int, float)):
            raise TypeError(f"taker_fee must be numeric, got {type(self.taker_fee)}")
        
        if self.taker_fee < 0 or self.taker_fee > 0.01:
            raise ValueError(f"taker_fee must be between 0 and 0.01 (0-1%), got {self.taker_fee}")
        
        # Validate slippage_model
        valid_models = ['fixed', 'volumetric', 'sqrt']
        if self.slippage_model not in valid_models:
            raise ValueError(
                f"Invalid slippage_model '{self.slippage_model}'. Must be one of {valid_models}"
            )
        
        # Validate slippage_basis_points
        if not isinstance(self.slippage_basis_points, (int, float)):
            raise TypeError(f"slippage_basis_points must be numeric, got {type(self.slippage_basis_points)}")
        
        if self.slippage_basis_points < 0 or self.slippage_basis_points > 100:
            raise ValueError(f"slippage_basis_points must be between 0 and 100, got {self.slippage_basis_points}")
        
        logger.debug(f"TransactionCosts validated: maker={self.maker_fee}, taker={self.taker_fee}")
    
    def calculate_slippage(self, price: float, volume: float = None) -> float:
        """Calculer le slippage selon le modèle"""
        if self.slippage_model == "fixed":
            return price * (self.slippage_basis_points / 10000)
        elif self.slippage_model == "sqrt" and volume:
            # Modèle sqrt pour impact de marché
            return price * (self.slippage_basis_points / 10000) * np.sqrt(volume / 1000)
        elif self.slippage_model == "volumetric" and volume:
            # Linéaire avec volume
            return price * (self.slippage_basis_points / 10000) * (volume / 1000)
        return price * (self.slippage_basis_points / 10000)


class CustomCommissionScheme(bt.CommInfoBase):
    """Schéma de commission personnalisé avec maker/taker fees"""
    
    params = (
        ('commission', 0.001),
        ('maker_fee', 0.0002),
        ('taker_fee', 0.0004),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """Calculer commission en fonction du type d'ordre"""
        # Simplification: on utilise taker fee par défaut
        # Dans une implémentation réelle, on détecterait le type d'ordre
        return abs(size) * price * self.p.taker_fee


class AdvancedBacktestEngine:
    """Moteur de backtesting de niveau institutionnel avec Backtrader"""
    
    def __init__(self, config: BacktestConfig, transaction_costs: TransactionCosts):
        # Validate config
        if config is None:
            raise ValueError("config cannot be None")
        
        if not isinstance(config, BacktestConfig):
            raise TypeError(f"config must be BacktestConfig, got {type(config)}")
        
        # Validate transaction_costs
        if transaction_costs is None:
            raise ValueError("transaction_costs cannot be None")
        
        if not isinstance(transaction_costs, TransactionCosts):
            raise TypeError(f"transaction_costs must be TransactionCosts, got {type(transaction_costs)}")
        
        self.config = config
        self.costs = transaction_costs
        self.cerebro = bt.Cerebro()
        self.data_manager = UnifiedDataManager()
        self.results = None
        self.trade_list = []
        
        logger.info(f"AdvancedBacktestEngine initialized for {config.strategy_name}")
        
    def _df_to_backtrader_feed(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """Convertir DataFrame en feed Backtrader"""
        # Validate DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Check minimum length
        if len(df) < 50:
            logger.warning(f"DataFrame has only {len(df)} rows. Recommend at least 50 for meaningful backtest.")
        
        # S'assurer que l'index est datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Renommer les colonnes au format Backtrader
        df_bt = df.copy()
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Vérifier et renommer
        for old, new in column_mapping.items():
            if old in df_bt.columns and old != new:
                df_bt.rename(columns={old: new}, inplace=True)
        
        # Créer le feed
        data = bt.feeds.PandasData(
            dataname=df_bt,
            datetime=None,  # Utilise l'index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        return data
        
    def setup(self):
        """Configurer le moteur Backtrader"""
        # 1. Ajouter la stratégie
        strategy_config = StrategyConfig(
            name=self.config.strategy_name,
            parameters=self.config.strategy_params,
            timeframe=self.config.timeframe,
            symbols=self.config.symbols,
            capital=self.config.initial_capital,
            framework=StrategyFramework.BACKTRADER
        )
        
        try:
            strategy_class = StrategyFactory.create(
                self.config.strategy_name, 
                strategy_config
            )
            self.cerebro.addstrategy(strategy_class)
        except Exception as e:
            print(f"⚠️  Erreur lors de la création de la stratégie: {e}")
            print(f"   Utilisation d'une stratégie de test simple")
            # Fallback vers une stratégie simple pour les tests
            self.cerebro.addstrategy(bt.Strategy)
        
        # 2. Ajouter les données
        for symbol in self.config.symbols:
            try:
                df = self.data_manager.get_data(
                    symbol, 
                    self.config.start_date, 
                    self.config.end_date
                )
                
                if df is not None and len(df) > 0:
                    data = self._df_to_backtrader_feed(df)
                    self.cerebro.adddata(data, name=symbol)
                    print(f"✅ Données chargées pour {symbol}: {len(df)} barres")
                else:
                    print(f"⚠️  Pas de données pour {symbol}")
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement de {symbol}: {e}")
        
        # 3. Configuration broker
        self.cerebro.broker.setcash(self.config.initial_capital)
        
        # Commission personnalisée avec maker/taker fees
        commission_scheme = CustomCommissionScheme(
            maker_fee=self.costs.maker_fee,
            taker_fee=self.costs.taker_fee
        )
        self.cerebro.broker.addcommissioninfo(commission_scheme)
        
        # Slippage
        if self.config.slippage_percent > 0:
            self.cerebro.broker.set_slippage_perc(
                self.config.slippage_percent,
                slip_open=True,
                slip_match=True
            )
        
        # 4. Ajouter analyseurs institutionnels
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.NoTimeFrame)
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Variability-Weighted Return
        
        # 5. Observers pour visualisation
        self.cerebro.addobserver(bt.observers.Broker)
        self.cerebro.addobserver(bt.observers.Trades)
        self.cerebro.addobserver(bt.observers.BuySell)
        self.cerebro.addobserver(bt.observers.DrawDown)
        
    def run(self) -> Dict[str, Any]:
        """Exécuter le backtest"""
        self.setup()
        
        print("\n" + "="*60)
        print("🚀 DÉMARRAGE DU BACKTEST AVANCÉ")
        print("="*60)
        print(f"💰 Capital initial: ${self.config.initial_capital:,.2f}")
        print(f"📊 Stratégie: {self.config.strategy_name}")
        print(f"📈 Symboles: {', '.join(self.config.symbols)}")
        print(f"⏰ Période: {self.config.start_date.date()} → {self.config.end_date.date()}")
        print(f"💸 Commission taker: {self.costs.taker_fee*100:.3f}%")
        print(f"📉 Slippage: {self.costs.slippage_basis_points} bps")
        print("="*60 + "\n")
        
        # Exécuter
        self.results = self.cerebro.run()
        
        # Résultats finaux
        final_value = self.cerebro.broker.getvalue()
        pnl = final_value - self.config.initial_capital
        pnl_pct = (pnl / self.config.initial_capital) * 100
        
        print("\n" + "="*60)
        print("📊 RÉSULTATS DU BACKTEST")
        print("="*60)
        print(f"💵 Capital final: ${final_value:,.2f}")
        print(f"📈 P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print("="*60 + "\n")
        
        return self.extract_metrics()
    
    def extract_metrics(self) -> Dict[str, Any]:
        """Extraire toutes les métriques institutionnelles"""
        if not self.results:
            return {}
            
        strat = self.results[0]
        
        # Valeurs de base
        final_value = self.cerebro.broker.getvalue()
        pnl = final_value - self.config.initial_capital
        pnl_pct = (pnl / self.config.initial_capital) * 100
        
        # Analyseur Sharpe
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe_analysis.get('sharperatio', None)
        if sharpe_ratio is None or np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0
        
        # Analyseur DrawDown
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0)
        max_dd_money = dd_analysis.get('max', {}).get('moneydown', 0)
        
        # Analyseur Trades
        trades_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trades_analysis.get('total', {}).get('total', 0)
        won_trades = trades_analysis.get('won', {}).get('total', 0)
        lost_trades = trades_analysis.get('lost', {}).get('total', 0)
        
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Gains/Pertes moyens
        avg_win = trades_analysis.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = trades_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
        
        # Profit factor
        gross_profit = trades_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(trades_analysis.get('lost', {}).get('pnl', {}).get('total', 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # SQN (System Quality Number)
        sqn_analysis = strat.analyzers.sqn.get_analysis()
        sqn = sqn_analysis.get('sqn', 0)
        
        # VWR (Variability-Weighted Return)
        vwr_analysis = strat.analyzers.vwr.get_analysis()
        vwr = vwr_analysis.get('vwr', 0)
        
        metrics = {
            # Performance
            'initial_capital': self.config.initial_capital,
            'final_value': final_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'total_return': pnl_pct / 100,
            
            # Risque
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_money': max_dd_money,
            
            # Trading
            'total_trades': total_trades,
            'won_trades': won_trades,
            'lost_trades': lost_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            
            # Qualité
            'sqn': sqn,
            'vwr': vwr,
            
            # Configuration
            'symbols': self.config.symbols,
            'timeframe': self.config.timeframe,
            'strategy': self.config.strategy_name,
            'parameters': self.config.strategy_params,
        }
        
        return metrics
    
    def plot(self, style: str = 'candlestick', **kwargs):
        """Générer les graphiques du backtest"""
        try:
            self.cerebro.plot(style=style, **kwargs)
        except Exception as e:
            print(f"⚠️  Erreur lors de la génération du graphique: {e}")
            print("   Les graphiques nécessitent un environnement avec display")
    
    def get_trades_list(self) -> List[Dict[str, Any]]:
        """Récupérer la liste détaillée des trades"""
        if not self.results:
            return []
        
        strat = self.results[0]
        trades_analysis = strat.analyzers.trades.get_analysis()
        
        # Note: Pour obtenir les trades individuels, il faudrait
        # implémenter un observer personnalisé dans la stratégie
        return []
