"""
EMA Crossover Strategy Adapter
Adapts EMA crossover strategy for Nautilus Trader and Backtrader
"""
from typing import Dict
from src.adapters.base_strategy_adapter import BaseStrategyAdapter, StrategyConfig

# Nautilus imports
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder

# Technical analysis
import talib
import numpy as np

# Backtrader imports
import backtrader as bt


class NautilusEMAStrategy(Strategy):
    """EMA Crossover Strategy for Nautilus Trader"""
    
    def __init__(self, config):
        super().__init__(config)
        self.fast_period = config.parameters['fast_period']
        self.slow_period = config.parameters['slow_period']
        self.instrument_id = None
        self.bar_type = None
        self.prices = []  # Store price history for talib
        self.position_size = config.capital * config.risk_params['max_position_size']
        
    def on_start(self):
        """Called when strategy starts"""
        # Get first symbol from config
        symbol = self.config.symbols[0].replace('/', '')  # BTC/USDT -> BTCUSDT
        self.instrument_id = InstrumentId.from_str(f"{symbol}.BINANCE")
        
        # Create bar type for subscription
        self.bar_type = BarType.from_str(
            f"{symbol}.BINANCE-{self.config.timeframe}-LAST"
        )
        
        # Subscribe to bars
        self.subscribe_bars(self.bar_type)
        
        self.log.info(f"EMA Strategy started: fast={self.fast_period}, slow={self.slow_period}")
    
    def on_bar(self, bar: Bar):
        """Called on each bar update"""
        # Collect price history
        self.prices.append(bar.close.as_double())
        
        # Wait for enough data
        if len(self.prices) < self.slow_period:
            return
        
        # Calculate EMAs using talib
        prices_array = np.array(self.prices[-self.slow_period:])
        ema_fast = talib.EMA(prices_array, timeperiod=self.fast_period)[-1]
        ema_slow = talib.EMA(prices_array, timeperiod=self.slow_period)[-1]
        
        # Get current position
        position = self.cache.position(self.instrument_id)
        
        # Trading logic: Buy when fast crosses above slow, Sell when fast crosses below slow
        if ema_fast > ema_slow and (position is None or position.is_closed):
            # Buy signal
            self.log.info(f"BUY signal: fast={ema_fast:.2f} > slow={ema_slow:.2f}")
            self._enter_long()
            
        elif ema_fast < ema_slow and position and position.is_long:
            # Sell signal
            self.log.info(f"SELL signal: fast={ema_fast:.2f} < slow={ema_slow:.2f}")
            self._exit_position()
    
    def _enter_long(self):
        """Enter long position"""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.position_size
        )
        self.submit_order(order)
    
    def _exit_position(self):
        """Exit current position"""
        position = self.cache.position(self.instrument_id)
        if position:
            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=position.quantity
            )
            self.submit_order(order)
    
    def on_stop(self):
        """Called when strategy stops"""
        self.log.info("EMA Strategy stopped")


class BacktraderEMAStrategy(bt.Strategy):
    """EMA Crossover Strategy for Backtrader"""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 50),
        ('max_position_size', 0.1),
        ('stop_loss_pct', 0.02),
    )
    
    def __init__(self):
        """Initialize strategy"""
        # Validate fast_period
        if not isinstance(self.params.fast_period, int):
            raise TypeError(f"fast_period must be int, got {type(self.params.fast_period)}")
        
        if self.params.fast_period <= 0 or self.params.fast_period > 200:
            raise ValueError(f"fast_period must be between 1 and 200, got {self.params.fast_period}")
        
        # Validate slow_period
        if not isinstance(self.params.slow_period, int):
            raise TypeError(f"slow_period must be int, got {type(self.params.slow_period)}")
        
        if self.params.slow_period <= 0 or self.params.slow_period > 200:
            raise ValueError(f"slow_period must be between 1 and 200, got {self.params.slow_period}")
        
        # Validate relationship
        if self.params.fast_period >= self.params.slow_period:
            raise ValueError(
                f"fast_period ({self.params.fast_period}) must be < slow_period ({self.params.slow_period})"
            )
        
        # Create EMA indicators
        self.fast_ema = bt.indicators.ExponentialMovingAverage(
            self.data.close, 
            period=self.params.fast_period
        )
        self.slow_ema = bt.indicators.ExponentialMovingAverage(
            self.data.close,
            period=self.params.slow_period
        )
        
        # Crossover indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)
        
        # Track orders
        self.order = None
        
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when trade is closed"""
        if trade.isclosed:
            self.log(f'TRADE PROFIT, Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')
    
    def next(self):
        """Called on each bar"""
        # Check if an order is pending
        if self.order:
            return
        
        # Check if we are in the market
        if not self.position:
            # Not in market, check for buy signal
            if self.crossover > 0:  # Fast crosses above slow
                # Calculate position size
                cash = self.broker.getcash()
                size = (cash * self.params.max_position_size) / self.data.close[0]
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}')
                self.order = self.buy(size=size)
        
        else:
            # In market, check for sell signal
            if self.crossover < 0:  # Fast crosses below slow
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}')
                self.order = self.sell(size=self.position.size)


class EMAAdapter(BaseStrategyAdapter):
    """Adapter for EMA Crossover Strategy"""
    
    def validate_parameters(self) -> None:
        """Validate EMA strategy parameters"""
        required = ['fast_period', 'slow_period']
        for param in required:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        fast = self.config.parameters['fast_period']
        slow = self.config.parameters['slow_period']
        
        if fast >= slow:
            raise ValueError(f"fast_period ({fast}) must be less than slow_period ({slow})")
        
        if fast < 2 or slow < 2:
            raise ValueError("Periods must be at least 2")
    
    def to_nautilus(self):
        """Convert to Nautilus Trader strategy"""
        return NautilusEMAStrategy(self.config)
    
    def to_backtrader(self):
        """Convert to Backtrader strategy class"""
        # Create a configured strategy class
        class ConfiguredEMAStrategy(BacktraderEMAStrategy):
            params = (
                ('fast_period', self.config.parameters['fast_period']),
                ('slow_period', self.config.parameters['slow_period']),
                ('max_position_size', self.config.risk_params['max_position_size']),
                ('stop_loss_pct', self.config.risk_params['stop_loss_pct']),
            )
        
        return ConfiguredEMAStrategy
    
    def get_parameter_space(self) -> Dict[str, tuple]:
        """Get optimization parameter space"""
        return {
            'fast_period': (5, 50),
            'slow_period': (20, 200)
        }
